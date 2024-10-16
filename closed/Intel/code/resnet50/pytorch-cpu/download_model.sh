#!/bin/bash

export MODEL_DIR=/model
wget --no-check-certificate https://zenodo.org/record/4588417/files/resnet50-19c8e357.pth -O "${MODEL_DIR}/resnet50-fp32-model.pth"
