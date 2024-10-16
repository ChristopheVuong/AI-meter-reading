# AI-meter-reading

## Context

As a provider of drinking water to millions of consumers, SUEZ needs to know the exact volume consumed by each client (ie the index of his meter). A modern solution to this problem is telemetering whereby the meter transmits automatically the daily index to our servers. This is already deployed to millions of meters, but there are still some contracts where our operators have to visit the meters once a year, sometimes more. This often involves arranging a meeting with the client when the meters are on private property, which can prove difficult (think about secondary housing) The goal of this challenge is to simplify the process by allowing the client to do the reading himself if it is more convenient: She could just take a picture of the meter, upload it to our servers whereupon a Machine Learning algorithm would validate it and read the digits to get the index.

Prototypes of this projects already exist but require the client to send the picture through email to our service center, which will analyze it and reply several hours later, making any feedback on the picture quality very difficult.

For this challenge, we'll assume that every image represents a meter with an index that can be read by a human.

The goal of this challenge is to design an algorithm reading the consumption index from a valid picture of a meter.

See https://challengedata.ens.fr/challenges/30

## Preparation

```
git clone git@github.com:ChristopheVuong/AI-meter-reading.git
cd AI-reading_meter
pip install -r requirements.txt
```

UNDER COURSE

