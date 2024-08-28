#!/bin/bash

export SCENARIO="${SCENARIO:-Offline}"
export ACCURACY="${ACCURACY:-false}"
export DTYPE="${DTYPE:-int4}"

source default.conf
export SYSTEM="${SYSTEM:-${SYSTEM_DEFAULT}}"
export WORKLOAD="${WORKLOAD:-${WORKLOAD_DEFAULT}}"

export DATA_DIR=/data
export MODEL_DIR=/model
export LOG_DIR="${LOG_DIR:-/logs}"
export RESULTS_DIR=${LOG_DIR}/results/${SYSTEM}/${WORKLOAD}/${SCENARIO}

if [ "${DTYPE}" == "int4" ]; then
        if ! [ -f "/model/gpt-j-checkpoint-final-q4-j-int8-pc.bin" ]; then
                echo "Quantized model file does not exist. Calibrating gpt-j-checkpoint now."
                bash run_calibration.sh
        fi
fi

bash run_clean.sh

if [ "$SCENARIO" == "Server" ]; then
        if [ "$ACCURACY" == "true" ]; then
                echo "Run GPT-J (${SCENARIO} Accuracy)."
                SCENARIO=Server MODE=Accuracy     WORKERS_PER_PROC=1 BATCH_SIZE=4  bash run_inference.sh
        else
                echo "Run GPT-J (${SCENARIO} Performance)."
                SCENARIO=Server MODE=Performance  WORKERS_PER_PROC=1 BATCH_SIZE=4  bash run_inference.sh
        fi
else
        if [ "$ACCURACY" == "true" ]; then
                echo "Run GPT-J (${SCENARIO} Accuracy)."
                SCENARIO=Offline MODE=Accuracy    WORKERS_PER_PROC=4 BATCH_SIZE=12 bash run_inference.sh
        else
                echo "Run GPT-J (${SCENARIO} Performance)."
                SCENARIO=Offline MODE=Performance WORKERS_PER_PROC=4 BATCH_SIZE=12 bash run_inference.sh
        fi
fi

if [ "$ACCURACY" == "true" ]; then
        OUTPUT_DIR=${RESULTS_DIR}/accuracy
	mkdir -p ${OUTPUT_DIR}
	cd $(ls -td -- *-output-$(hostname)*/ | head -n 1)
	mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt accuracy.txt ${OUTPUT_DIR}
else
        OUTPUT_PREFIX=${RESULTS_DIR}/performance/run
        RUN_COUNT=1
        while [ -d "${OUTPUT_PREFIX}_${RUN_COUNT}" ]; do
                RUN_COUNT=$(( RUN_COUNT + 1 ))
        done
        OUTPUT_DIR=${OUTPUT_PREFIX}_${RUN_COUNT}
        mkdir -p ${OUTPUT_DIR}
	cd $(ls -td -- *-output-$(hostname)*/ | head -n 1)
        mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt ${OUTPUT_DIR}
fi
