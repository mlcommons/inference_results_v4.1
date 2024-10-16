#!/bin/bash

export DATA_DIR=/data
cd ${DATA_DIR}
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
tar -xvf ILSVRC2012_img_val.tar
rm ILSVRC2012_img_val.tar
wget https://raw.githubusercontent.com/mlcommons/inference_results_v4.0/main/closed/Intel/code/resnet50/pytorch-cpu/val_data/val_map.txt
