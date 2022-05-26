import random
import typing
import asyncio
import os
import time
import sys

import discord
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from functools import partial, wraps
from dotenv import load_dotenv

from flax.jax_utils import replicate
from flax.training.common_utils import shard_prng_key

import jax
import numpy as np
import jax.numpy as jnp

from io import BytesIO
from PIL import Image

load_dotenv()

if os.getenv('TOKEN') is None:
  print("no token provided", file=sys.stderr)
  exit(1)

os.environ['WANDB_MODE']="disabled"

# type used for computation - use bfloat16 on TPU's
dtype = jnp.bfloat16 if jax.local_device_count() == 8 else jnp.float32

# TODO: fix issue with bfloat16
dtype = jnp.float32

# dalle-mini
DALLE_MODEL = "dalle-mini/dalle-mini/wzoooa1c:latest"  # can be wandb artifact or 🤗 Hub or local folder or google bucket
# DALLE_MODEL = 'dalle-mini/dalle-mini/mega-1:latest' # uncomment this line to use DALL-E Mega. Warning: requires significantly more storage and GPU RAM
DALLE_COMMIT_ID = None

# VQGAN model
VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

gen_top_k = None
gen_top_p = 0.9
temperature = None
cond_scale = 3.0

model = DalleBart.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)
vqgan = VQModel.from_pretrained(VQGAN_REPO, revision=VQGAN_COMMIT_ID)

# convert model parameters for inference if requested
if dtype == jnp.bfloat16:
  model.params = model.to_bf16(model.params)

model._params = replicate(model.params)
vqgan._params = replicate(vqgan.params)

processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)

def to_thread(func: typing.Callable) -> typing.Coroutine:
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)
    return wrapper

# model inference
@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
def p_generate(tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale):
  return model.generate(
    **tokenized_prompt,
    prng_key=key,
    params=params,
    top_k=top_k,
    top_p=top_p,
    temperature=temperature,
    condition_scale=condition_scale,
  )

# decode images
@partial(jax.pmap, axis_name="batch")
def p_decode(indices, params):
  return vqgan.decode_code(indices, params=params)

def tokenize_prompt(prompt: str):
  tokenized_prompt = processor([prompt])
  return replicate(tokenized_prompt)

@to_thread
def generate_images(prompt:str, num_predictions: int):
  print('Generating: {0}'.format(prompt))

  tokenized_prompt = tokenize_prompt(prompt)
  
  # create a random key
  seed = random.randint(0, 2**32 - 1)
  key = jax.random.PRNGKey(seed)

  # generate images
  images = []
  for i in range(num_predictions // jax.device_count()):
    # get a new key
    key, subkey = jax.random.split(key)
    
    # generate images
    encoded_images = p_generate(tokenized_prompt, shard_prng_key(subkey),
        model.params,gen_top_k, gen_top_p, temperature, cond_scale,
    )
    
    # remove BOS
    encoded_images = encoded_images.sequences[..., 1:]

    # decode images
    decoded_images = p_decode(encoded_images, vqgan.params)
    decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
    for img in decoded_images:
      images.append(Image.fromarray(np.asarray(img * 255, dtype=np.uint8)))
        
  return images

ratelimit = {}

class GenThot(discord.Client):
  async def on_ready(self):
    print('Logged on as {0}!'.format(self.user))

  async def on_message(self, message: discord.Message):
    text: str = message.content
  
    if text.startswith('*gen '):
      start = time.time()
      await message.channel.trigger_typing();

      if message.author.id in ratelimit and start - ratelimit[message.author.id] < 15:
        await message.reply("wait 15 seconds")
        return

      prompt = text[5:].strip()

      if len(prompt) > 0:
        ratelimit[message.author.id] = start
        generated_imgs = await generate_images(prompt, 1)

        for img in generated_imgs:
            end = time.time()
            ratelimit[message.author.id] = end

            with BytesIO() as buffered:
              img.save(buffered, format="JPEG")
              buffered.seek(0)
              await message.reply("took {:.2f} seconds".format(end - start), file=discord.File(buffered, "generated.jpg"))
      else:
        await message.reply("soz u neet 2 specify")

client = GenThot()
client.run(os.getenv('TOKEN'))