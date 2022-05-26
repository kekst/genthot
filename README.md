# genthot

![genthot](/img.png "lol hi")

discord bot based on dalle-mini that generates funny images when told to

https://github.com/borisdayma/dalle-mini

wanna play with it? we have it running in https://discord.gg/technology

## install

assuming you have python 3.9 or better

```
pip install -r requirements.txt
```

## config

create a file called `.env` with this:

```
TOKEN=YOUR_BOT_TOKEN
```

replace `YOUR_BOT_TOKEN` with the real deal, y'know

## start

assuming you have a noncucked distro:

```
python bot.py
```

on cucked distros run:

```
python3 bot.py
```

## interacting

once invited the bot will pass anything after `*gen` as the prompt
