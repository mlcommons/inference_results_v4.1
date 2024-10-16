#!/bin/bash

export MODEL_DIR=/model
wget --no-check-certificate 'https://zenodo.org/record/6617981/files/resnext50_32x4d_fpn.pth' -O "${MODEL_DIR}/retinanet-model.pth"
