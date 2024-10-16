#!/bin/bash

export DATA_DIR=/data
cd /workspace
python download-dataset.py --split validation --output-dir ${DATA_DIR}
python download-calibration-dataset.py --calibration-list-file calibration-list.txt --output-dir ${DATA_DIR}
