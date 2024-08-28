#!/bin/bash

export MODEL_DIR=/model
export DATA_DIR=/data

export CRITEO_DIR=${DATA_DIR}/criteo_1tb
export RAW_DATASET=${CRITEO_DIR}/raw_input_dataset_dir
export TEMP_FILES=${CRITEO_DIR}/temp_intermediate_files_dir

# Creates: /model/dlrm-multihot-pytorch.pt
numactl -C 56-111 -m 1 python python/dump_torch_model.py \
        --model-path=${MODEL_DIR} \
        --dataset-path=${DATA_DIR}

export MODEL_PATH=${MODEL_DIR}/dlrm-multihot-pytorch.pt

numactl -C 56-111 -m 1 python python/calibration.py \
        --max-batchsize=65536 \
        --model-path=${MODEL_PATH} \
        --dataset-path=${DATA_DIR} \
        --use-int8 --calibration

mv dlrm_int8.pt ${MODEL_DIR}/
