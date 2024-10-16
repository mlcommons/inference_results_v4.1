#!/bin/bash

export SCENARIO="${SCENARIO:-Offline}"
export ACCURACY="${ACCURACY:-false}"
export DTYPE="${DTYPE:-int8}"

source default.conf
export SYSTEM="${SYSTEM:-${SYSTEM_DEFAULT}}"
export WORKLOAD="${WORKLOAD:-${WORKLOAD_DEFAULT}}"

export DATA_DIR=/data
export MODEL_DIR=/model
export LOG_DIR="${LOG_DIR:-/logs}"
export COMPLIANCE="${COMPLIANCE}"
export RESULTS_DIR=${LOG_DIR}/results/${SYSTEM}/${WORKLOAD}/${SCENARIO}

export ENV_DEPS_DIR=/workspace/retinanet-env
export MODEL_CHECKPOINT=/model/retinanet-model.pth
export MODEL_PATH=/model/retinanet-int8-model.pth

if [ "${DTYPE}" == "int8" ]; then
        if ! [ -f "${MODEL_PATH}" ]; then
                echo "The model has not been quantized as INT8 yet.  Performing this one-time calibration now."
		export MODEL_CHECKPOINT=/model/retinanet-model.pth
		export CALIBRATION_DIR=/model/openimages-calibrated
		export CALIBRATION_DATA_DIR=${CALIBRATION_DIR}/train/data
                export CALIBRATION_ANNOTATIONS=${CALIBRATION_DIR}/annotations/openimages-mlperf-calibration.json
		bash openimages_calibration_mlperf.sh --dataset-path ${CALIBRATION_DIR}
                bash run_calibration.sh
		echo "INT8 calibration complete."
        fi
fi

bash run_clean.sh

if [ "$SCENARIO" == "Server" ]; then
        if [ "$ACCURACY" == "true" ]; then
                echo "Run ${MODEL_DEFAULT} (${SCENARIO} Accuracy)."
                bash run_server_accuracy.sh
        else
                echo "Run ${MODEL_DEFAULT} (${SCENARIO} Performance)."
                bash run_server.sh
        fi
else
        if [ "$ACCURACY" == "true" ]; then
                echo "Run ${MODEL_DEFAULT} (${SCENARIO} Accuracy)."
                bash run_offline_accuracy.sh
        else
                echo "Run ${MODEL_DEFAULT} (${SCENARIO} Performance)."
                bash run_offline.sh
        fi
fi

if [ -z "${COMPLIANCE}" ]; then
    # This is a genuine run, not compliance
    if [ "$ACCURACY" == "true" ]; then
            OUTPUT_DIR=${RESULTS_DIR}/accuracy
            mkdir -p ${OUTPUT_DIR}
            mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt accuracy.txt ${OUTPUT_DIR}
    else
            OUTPUT_DIR=${RESULTS_DIR}/performance/run_1
            mkdir -p ${OUTPUT_DIR}
            mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt ${OUTPUT_DIR}
    fi
else
    # This is a compliance run
    echo "Launching compliance test: ${COMPLIANCE}"
    OUTPUT_DIR=${LOG_DIR}
    mkdir -p ${OUTPUT_DIR}
    mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt ${OUTPUT_DIR}
fi
