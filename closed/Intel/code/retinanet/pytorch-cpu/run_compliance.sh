#!/bin/bash

source default.conf
export SYSTEM="${SYSTEM:-${SYSTEM_DEFAULT}}"
export WORKLOAD="${WORKLOAD:-${WORKLOAD_DEFAULT}}"
export MODEL="${MODEL:-${MODEL_DEFAULT}}"
export COMPLIANCE_TESTS="${COMPLIANCE_TESTS:-${COMPLIANCE_TESTS_DEFAULT}}"

export COMPLIANCE_SUITE_DIR=/workspace/inference/compliance/nvidia
export LOG_DIR="${LOG_DIR:-/logs}"
export RESULTS_DIR=/${LOG_DIR}/results
export COMPLIANCE_LOGS=${LOG_DIR}/tmp
export COMPLIANCE_OUTPUT=${LOG_DIR}/compliance

for TEST in ${COMPLIANCE_TESTS[@]}
do
        echo "Running compliance ${TEST} ..."

        if [ "$TEST" == "TEST01" ]; then
                cp ${COMPLIANCE_SUITE_DIR}/${TEST}/${MODEL}/audit.config .
        else
                cp ${COMPLIANCE_SUITE_DIR}/${TEST}/audit.config .
        fi

        mkdir -p ${COMPLIANCE_LOGS}

        for SCENARIO in Offline Server; do
            RESULTS=${RESULTS_DIR}/${SYSTEM}/${WORKLOAD}/${SCENARIO}
            COMPLIANCE=${COMPLIANCE_LOGS}/${TEST}/${SCENARIO}
            OUTPUT=${COMPLIANCE_OUTPUT}/${SYSTEM}/${WORKLOAD}/${SCENARIO}
            LOG_DIR=${COMPLIANCE} COMPLIANCE=${TEST} SCENARIO=${SCENARIO} ACCURACY=false bash run_mlperf.sh
            python ${COMPLIANCE_SUITE_DIR}/${TEST}/run_verification.py -r ${RESULTS} -c ${COMPLIANCE} -o ${OUTPUT}
        done

        rm -r ${COMPLIANCE_LOGS}
        rm ./audit.config
done
