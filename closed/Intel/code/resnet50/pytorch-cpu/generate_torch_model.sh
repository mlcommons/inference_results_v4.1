#!/bin/bash

export MODEL_DIR="${MODEL_DIR:-/model}"

export DATA_CAL_DIR=${MODEL_DIR}/calibration_dataset
export CHECKPOINT=${MODEL_DIR}/resnet50-fp32-model.pth

export ARGS="--batch-size 1 --data-path-cal ${DATA_CAL_DIR} --checkpoint-path ${CHECKPOINT} --save-dir ${MODEL_DIR} --calibrate-start-partition --calibrate-end-partition --calibrate-full-weights --save-full-weights --channels-last --massage"

numactl python -u main.py ${ARGS}

echo "Generating binary scales data for kernel backbone"

python src/ckernels/scripts/make_qinfo.py --infile src/ckernels/scripts/resnet50-int8-scales.json --outfile backbone_data_256.cpp --batchsize 256
python src/ckernels/scripts/make_qinfo.py --infile src/ckernels/scripts/resnet50-int8-scales.json --outfile backbone_data_8.cpp --batchsize 8
python src/ckernels/scripts/make_qinfo.py --infile src/ckernels/scripts/resnet50-int8-scales.json --outfile backbone_data_4.cpp --batchsize 4
mv backbone_data_* src/ckernels/src/kernel_rn50
