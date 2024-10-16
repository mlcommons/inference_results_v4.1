#!/bin/bash

export DATA_DIR=/data
cd /workspace/retinanet-env/mlperf_inference/vision/classification_and_detection/tools
bash openimages_mlperf.sh --dataset-path ${DATA_DIR}
