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

export BERT_FILE=${MODEL_DIR}/bert.pt
export SQUAD_FILE=${MODEL_DIR}/squad.pt

if [ "${DTYPE}" == "int8" ]; then
        if ! [ -f "${BERT_FILE}" ] || ! [ -f "${SQUAD_FILE}" ]; then
                echo "Needed model file does not exist. Calibrating BERT file now."
                DATA_DIR=/data MODEL_DIR=/model bash run_calibration.sh
        fi
fi

bash run_clean.sh

if [ "$SCENARIO" == "Server" ]; then
        if [ "$ACCURACY" == "true" ]; then
                echo "Run ${MODEL_DEFAULT} (${SCENARIO} Accuracy)."
                bash run_server.sh --accuracy
        else
                echo "Run ${MODEL_DEFAULT} (${SCENARIO} Performance)."
                bash run_server.sh
        fi
else
        if [ "$ACCURACY" == "true" ]; then
                echo "Run ${MODEL_DEFAULT} (${SCENARIO} Accuracy)."
                bash run.sh --accuracy
        else
                echo "Run ${MODEL_DEFAULT} (${SCENARIO} Performance)."
                bash run.sh
        fi
fi

if [ -z "${COMPLIANCE}" ]; then
    # This is a genuine run, not compliance
    if [ "$ACCURACY" == "true" ]; then
            OUTPUT_DIR=${RESULTS_DIR}/accuracy
            mkdir -p ${OUTPUT_DIR}
	    cd test_log
            mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt accuracy.txt ${OUTPUT_DIR}
    else
            OUTPUT_DIR=${RESULTS_DIR}/performance/run_1
            mkdir -p ${OUTPUT_DIR}
	    cd test_log
            mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt ${OUTPUT_DIR}
    fi
else
    # This is a compliance run
    echo "Launching compliance test: ${COMPLIANCE}"
    OUTPUT_DIR=${LOG_DIR}
    mkdir -p ${OUTPUT_DIR}
    cd test_log
    mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt ${OUTPUT_DIR}
fi
