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

export RN50_START=${MODEL_DIR}/resnet50-start-int8-model.pth
export RN50_END=${MODEL_DIR}/resnet50-end-int8-model.pth
export RN50_FULL=${MODEL_DIR}/resnet50-full.pth

if [ "${DTYPE}" == "int8" ]; then
        if ! [ -f "${RN50_START}" ] || ! [ -f "${RN50_END}" ] || ! [ -f "${RN50_FULL}" ]; then
                echo "Needed model file does not exist. Calibrating rn50.pt file now."
                bash run_calibration.sh
        fi
fi

bash run_clean.sh

if [ "$SCENARIO" == "Server" ]; then
        if [ "$ACCURACY" == "true" ]; then
                echo "Run ${MODEL_DEFAULT} (${SCENARIO} Accuracy)."
                bash run_server_accuracy.sh
		mv server_accuracy.txt accuracy.txt
        else
                echo "Run ${MODEL_DEFAULT} (${SCENARIO} Performance)."
                bash run_server.sh
        fi
else
        if [ "$ACCURACY" == "true" ]; then
                echo "Run ${MODEL_DEFAULT} (${SCENARIO} Accuracy)."
                bash run_offline_accuracy.sh 256
		mv offline_accuracy.txt accuracy.txt
        else
                echo "Run ${MODEL_DEFAULT} (${SCENARIO} Performance)."
                bash run_offline.sh 256
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
