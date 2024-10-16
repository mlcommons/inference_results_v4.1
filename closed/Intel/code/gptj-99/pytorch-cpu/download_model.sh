#!/bin/bash

export MODEL_DIR=/model
cd ${MODEL_DIR}
wget https://cloud.mlcommons.org/index.php/s/QAZ2oM94MkFtbQx/download -O "${MODEL_DIR}/gpt-j-checkpoint.zip"
unzip -j gpt-j-checkpoint.zip "gpt-j/checkpoint-final/*" -d "${MODEL_DIR}/gpt-j-checkpoint"
rm -f gpt-j-checkpoint.zip
