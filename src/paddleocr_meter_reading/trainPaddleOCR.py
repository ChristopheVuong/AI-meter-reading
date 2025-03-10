# train and store in fine-tuned models in the project folder

# establish a symbolic link to the tools directory
# find the subdirectory where tools/train.py is located in PaddleOCR
# how to access the actual training method or function
# def train_from_paddle_ocr programming
# from paddleocr import tools.train as ttrain

# argparse in python file as configsè=/rec/PP-OCRv3/en_PP-OCRv3_rec.yml -o Global.pretrained_model=en_PP-OCRv3_rec_train/best_accuracy
# add tests related to PaddleOCR tests

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
# modules in the PaddleOCR directory
from paddleocr.ppocr.utils.utility import set_seed
import paddleocr.tools.program as program
from paddleocr.tools.train import main

# environment variables for the train directory, configs, etc
# __dir__ = os.path.dirname(os.path.abspath(__file__))
# # sys.path.append(__dir__) no need here because inside "nested folder"
# sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))
# making it current for the data_dir in YAML file
# issue is for the import or run the command (basically we only want to import main from train.py without any other side effects)
# no need to expose ppocr to the user

# tools look for the one in paddleocr




# rewrite build dataloader

# # GPU training Support single card and multi-card training
# Training icdar15 English data and The training log will be automatically saved as train.log under "{save_model_dir}"

#specify the single card training(Long training time, not recommended)
# python3 tools/train.py -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml -o Global.pretrained_model=en_PP-OCRv3_rec_train/best_accuracy
# Ok for the path here because of .toml file
#specify the card number through --gpus
# python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml -o Global.pretrained_model=en_PP-OCRv3_rec_train/best_accuracy



if __name__ == "__main__":
    config, device, logger, vdl_writer = program.preprocess(is_train=True)
    seed = config["Global"]["seed"] if "seed" in config["Global"] else 1024
    set_seed(seed)
    main(config, device, logger, vdl_writer, seed)
    # test_reader(config, device, logger)
    # USE THE SAME COMMAND LINE AS RECOMMENDED IN THE PADDLEOCR DOCUMENTATION
    # python3 tools/train.py -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml -o Global.pretrained_model=en_PP-OCRv3_rec_train/best_accuracy
    # simply modify the config YAML file accordingly