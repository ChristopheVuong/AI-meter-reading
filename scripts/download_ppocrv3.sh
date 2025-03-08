#!/bin/bash
# cd PaddleOCR/
cd ../..
# Download the pre-trained model of en_PP-OCRv3
wget -P models/ https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar

if [ -f models/en_PP-OCRv3_rec_train.tar ]; then
    echo "Download successful."
else
    echo "Download failed."
    exit 1
fi
# Check if the model tar file exists before decompressing
if [ -f models/en_PP-OCRv3_rec_train.tar ]; then
    # Decompress model parameters
    cd models
    tar -xf en_PP-OCRv3_rec_train.tar && rm -rf en_PP-OCRv3_rec_train.tar
else
    echo "Model tar file not found."
    exit 1
fi