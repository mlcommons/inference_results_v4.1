#!/bin/bash

export DATA_DIR="${DATA_DIR:-/data}"
export MODEL_DIR="${MODEL_DIR:-/model}"

bash prepare_calibration_dataset.sh
bash generate_torch_model.sh
