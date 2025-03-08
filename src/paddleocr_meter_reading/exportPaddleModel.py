import os
import sys

# avoid bloating the system path with unnecessary directories
# This script aims at exporting the trained model to be used in the inference pipeline, but maybe serialized it by the same way as the training pipeline
# I think those scripts are not needed in the final version of the project
# except to have a logger

import argparse

from paddleocr.tools.program import load_config, merge_config, ArgsParser
from paddleocr.ppocr.utils.export_model import export


def main():
    FLAGS = ArgsParser().parse_args()
    config = load_config(FLAGS.config)
    config = merge_config(config, FLAGS.opt)
    # export model
    export(config)


if __name__ == "__main__":
    main()