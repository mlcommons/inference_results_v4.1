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

number_threads=`nproc --all`
export number_cores=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
export NUM_SOCKETS=`grep physical.id /proc/cpuinfo | sort -u | wc -l`
export CPUS_PER_SOCKET=$((number_cores/NUM_SOCKETS))

export CPUS_PER_PROCESS=${CPUS_PER_SOCKET}  # which determine how much processes will be used
                            # process-per-socket = CPUS_PER_SOCKET/CPUS_PER_PROCESS
export CPUS_PER_INSTANCE=2  # instance-per-process number=CPUS_PER_PROCESS/CPUS_PER_INSTANCE
                            # total-instance = instance-per-process * process-per-socket
export CPUS_FOR_LOADGEN=1   # number of cpus for loadgen
                            # finally used in our code is max(CPUS_FOR_LOADGEN, left cores for instances)
export BATCH_SIZE=200
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX

bash run_clean.sh
export TMP_DIR=${LOG_DIR}/run_tmp

if [ "$SCENARIO" == "Server" ]; then
        if [ "$ACCURACY" == "true" ]; then
                echo "Run ${MODEL_DEFAULT} (${SCENARIO} Accuracy)."
		bash run_main.sh server accuracy ${DTYPE}
        else
                echo "Run ${MODEL_DEFAULT} (${SCENARIO} Performance)."
                bash run_main.sh server ${DTYPE}
        fi
else
        if [ "$ACCURACY" == "true" ]; then
                echo "Run ${MODEL_DEFAULT} (${SCENARIO} Accuracy)."
                bash run_main.sh offline accuracy ${DTYPE}
        else
                echo "Run ${MODEL_DEFAULT} (${SCENARIO} Performance)."
                bash run_main.sh offline ${DTYPE}
        fi
fi

if [ -z "${COMPLIANCE}" ]; then
    # This is a genuine run, not compliance
    if [ "$ACCURACY" == "true" ]; then
            OUTPUT_DIR=${RESULTS_DIR}/accuracy
            mkdir -p ${OUTPUT_DIR}
	    cd ${TMP_DIR}
            mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt accuracy.txt ${OUTPUT_DIR}
    else
            OUTPUT_DIR=${RESULTS_DIR}/performance/run_1
            mkdir -p ${OUTPUT_DIR}
	    cd ${TMP_DIR}
            mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt ${OUTPUT_DIR}
    fi
    rm -r ${TMP_DIR}
else
    # This is a compliance run
    echo "Launching compliance test: ${COMPLIANCE}"
    OUTPUT_DIR=${LOG_DIR}
    mkdir -p ${OUTPUT_DIR}
    cd ${TMP_DIR}
    mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt ${OUTPUT_DIR}
    rm -r ${TMP_DIR}
fi
