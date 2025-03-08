<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
</p>


# AI-meter-reading

## Context

As a provider of drinking water to millions of consumers, SUEZ needs to know the exact volume consumed by each client (ie the index of his meter). A modern solution to this problem is telemetering whereby the meter transmits automatically the daily index to our servers. This is already deployed to millions of meters, but there are still some contracts where our operators have to visit the meters once a year, sometimes more. This often involves arranging a meeting with the client when the meters are on private property, which can prove difficult (think about secondary housing) The goal of this challenge is to simplify the process by allowing the client to do the reading himself if it is more convenient: She could just take a picture of the meter, upload it to our servers whereupon a Machine Learning algorithm would validate it and read the digits to get the index.

Prototypes of this projects already exist but require the client to send the picture through email to the service center, which will analyze it and reply several hours later, making any feedback on the picture quality very difficult.

For this challenge, we'll assume that every image represents a meter with an index that can be read by a human.

The goal of this challenge is to design an algorithm reading the consumption index from a valid picture of a meter.

See https://challengedata.ens.fr/challenges/30

## Preparation

We use Python.

```bash
git clone git@github.com:ChristopheVuong/AI-meter-reading.git
cd AI-reading_meter
```

<!-- pip install -r requirements.txt -->
<!-- Ensure you have Poetry -->

Powered with Poetry.
It is important to proceed with the command 
```bash 
poetry run [script]
``` 
because Poetry activates the virtual environment associated with your project. During this activation, Poetry ensures that the project root (the directory containing `pyproject.toml`) is added to the `PYTHONPATH`, a list of directories that Python searches for modules and packages.


## Data

The data consists of (around) 1000 annotated RGB pictures of meters (cyclometers). The images are anonymous. No addresses are exposed. However, since they come from Suez's clients, one should request permission to work with them. As such, please contact christophe.vuong108@gmail.com in order to get the authorization to get source data.

**The quality is quite heterogeneous as can be expected given that meters are often located underground.** The meters share a common shape but are not all identical, and some can be rotated. The index part consists of (up to) 8 rotating wheels to display the digits of the index: 5 white on black digits for the cubic meters, followed by 3 white on red (or red on white) digits for the liters. By construction, it can happen that the wheel are rotating precisely at the moment the picture is taken, but we'll make sure for this challenge that they are unambiguous.

For each of the pictures, a human annotated the index in cubic meters (truncated).

## OCR problem

It is an OCR problem. Is it two stage or not?

YOLO vs PaddleOCR text detection. For YOLO, we use the "proprietary" Ultralytics and Roboflow annotation tool which provides with various format for the coordinates of bounding boxes.

### Models 

The models used are pre-trained fine-tuned ones either YOLO, CRNN, PaddleOCR or Vision Transformers, or Qwen-VL.

### Datasets

Roboflow YOLO dataset structure looks like:

```
dataset/
|-- train/
| |-- images/
| |-- labels/
|-- val/
| |-- images/
| |-- labels/
|-- test/
| |-- images/
| |-- labels/
```

PaddleOCR dataset structure looks like:

## Solutions

There are two approaches.

### Fine-tuned YOLO for automatic annotation

- The fine-tuning process involves feeding your custom dataset into the model and adjusting its weights using backpropagation.
- The goal is to minimize the loss function, which measures the difference between the predicted bounding boxes and the ground truth annotations.

For license issue in case of commercial use, refer to forks od YOLOv4 Darknet or alternatives. 

**Resizing images**

### PaddleOCRv3

PP-OCRv3 is composed of three parts: detection, classification and recognition, all of which can be used independently. Each part has its own model trained with the PaddlePaddle framework. For those interested, model details can be found in this dedicated research article PP-OCRv3: More Attempts for the Improvement of Ultra Lightweight OCR System (Yanjun et al., 2022)[].

In order to train the model, run the shell script:
```bash
poetry run 
```

Then run the `train.py` script which looks up for the associated model argparse to get that.

Follow the steps in https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/ppocr/model_train/finetune.en.md


For Fine-tune based on the PaddleOCR model, 500 sheets are generally required to achieve good results. In order to get more data, proceed with basic image processing or transformation based on PIL and opencv. For example, the three modules of ImageFont, Image, ImageDraw in PIL write text into the background, opencv's rotating affine transformation, Gaussian filtering and so on.

Prepare data according to General Data: Used for training with datasets stored in text files (SimpleDataSet);

**Note: In the txt file, please use \t to separate the image path and the label. Using any other separator will cause errors during training.**

## Further considerations



### CI/CD

One can use Github Actions.
<!-- TO CONTINUE -->

```yml
# .github/workflows/ci.yml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Test
        run: pytest tests/
  deploy-api:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to AWS
        # ... deployment steps ...
```

UNDER COURSE


Project Open Source

Possible to run Github actions

Credits: Suez Team

