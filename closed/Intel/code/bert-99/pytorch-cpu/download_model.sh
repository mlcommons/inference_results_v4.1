#!/bin/bash

export MODEL_DIR=/model
cd ${MODEL_DIR}
git clone https://huggingface.co/bert-large-uncased
mv bert-large-uncased/* .
rm -rf bert-large-uncased
wget https://zenodo.org/record/4792496/files/pytorch_model.bin?download=1 -O pytorch_model.bin
