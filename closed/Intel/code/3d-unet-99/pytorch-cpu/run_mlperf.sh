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

export CALIBRATED_MODEL=/model/unet3d_jit_model.pt

if [ "${DTYPE}" == "int8" ]; then
        if ! [ -f "${CALIBRATED_MODEL}" ]; then
                echo "Needed model file does not exist. Calibrating unet3d_jit_model.pt file now."
                bash run_calibration.sh
        fi
fi

bash run_clean.sh

if [ "$ACCURACY" == "true" ]; then
        echo "Run ResNet-50 (${SCENARIO} Accuracy)."
	bash run.sh acc
else
        echo "Run ResNet-50 (${SCENARIO} Performance)."
	bash run.sh perf
fi

if [ -z "${COMPLIANCE}" ]; then
    # This is a genuine run, not compliance
    if [ "$ACCURACY" == "true" ]; then
            OUTPUT_DIR=${RESULTS_DIR}/accuracy
            mkdir -p ${OUTPUT_DIR}
	    cd output_logs
            mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt accuracy.txt ${OUTPUT_DIR}
    else
            OUTPUT_DIR=${RESULTS_DIR}/performance/run_1
            mkdir -p ${OUTPUT_DIR}
	    cd output_logs
            mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt ${OUTPUT_DIR}
    fi
else
    # This is a compliance run
    echo "Launching compliance test: ${COMPLIANCE}"
    OUTPUT_DIR=${LOG_DIR}
    mkdir -p ${OUTPUT_DIR}
    cd output_logs
    mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt ${OUTPUT_DIR}
fi
