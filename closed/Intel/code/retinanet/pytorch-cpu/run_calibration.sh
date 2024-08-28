#!/bin/bash

export MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-/model/retinanet-model.pth}"
export CALIBRATION_DIR="${CALIBRATION_DIR:-/model/openimages-calibrated}"
export CALIBRATION_DATA_DIR="${CALIBRATION_DATA_DIR:-${CALIBRATION_DIR}/calibration/data}"
export CALIBRATION_ANNOTATIONS="${CALIBRATION_ANNOTATIONS:-${CALIBRATION_DIR}/annotations/openimages-calibration-mlperf.json}"

if [ -z "${CALIBRATION_DATA_DIR}" ]; then
    echo "Path to dataset not set. Please set it:"
    echo "export CALIBRATION_DATA_DIR="
    exit 1
fi

if [ -z "${MODEL_CHECKPOINT}" ]; then
    echo "Path to trained checkpoint not set. Please set it:"
    echo "export MODEL_CHECKPOINT="
    exit 1
fi

if [ -z "${CALIBRATION_ANNOTATIONS}" ]; then
    echo "Path to annotations for calibration images not set. Please set it:"
    echo "export CALIBRATION_ANNOTATIONS="
    exit 1
fi

# Install missing torchvision (difficulty installing during build time)
export VISION_VERSION=8e078971b8aebdeb1746fea58851e3754f103053
echo "Building Torchvision to enable model calibration"
git clone https://github.com/pytorch/vision && \
cd vision && \
git config user.email "test@example.com" && \
git checkout ${VISION_VERSION} && \
python setup.py install && \
VISION_DIR=${PWD} && \
cd .. && \
rm -rf ${VISION_DIR}
echo "Torchvision build complete."

# Downloading calibration data
cd /workspace/retinanet-env/mlperf_inference/vision/classification_and_detection/tools
bash openimages_calibration_mlperf.sh --dataset-path ${CALIBRATION_DIR}

# Calibrating model (NUM_CLASSES=264)
cd /workspace
export ARGS="--calibrate --cal-iters 500 --precision int8 --num-classes 264 --batch-size 1 --quantized-weights int8-scales-264.json --data-path ${CALIBRATION_DATA_DIR} --annotation-file ${CALIBRATION_ANNOTATIONS} --num-iters 500 --checkpoint-path ${MODEL_CHECKPOINT} --save-trace-model --save-trace-model-path $( dirname ${MODEL_CHECKPOINT} )/retinanet-int8-model.pth"

python -u helpers/main.py ${ARGS}
