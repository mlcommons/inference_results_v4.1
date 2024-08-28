#!/bin/bash

source default.conf
export SYSTEM="${SYSTEM:-${SYSTEM_DEFAULT}}"
export WORKLOAD="${WORKLOAD:-${WORKLOAD_DEFAULT}}"
export IMPL="${IMPL:-${IMPL_DEFAULT}}"

export LOG_DIR=/logs
export CALIBRATION_DIR=${LOG_DIR}/calibration/${WORKLOAD}/${IMPL}
export CODE_DIR=${LOG_DIR}/code/${WORKLOAD}/${IMPL}
export MEASUREMENTS_DIR=${LOG_DIR}/measurements/${SYSTEM}/${WORKLOAD}
export SYSTEMS_DIR=${LOG_DIR}/systems

export CLEAR_CONTENT="${CLEAR_CONTENT:-false}"

if [ "${CLEAR_CONTENT}" == "true" ]; then
    rm -rf ${CALIBRATION_DIR}
    rm -rf ${CODE_DIR}
    rm -rf ${MEASUREMENTS_DIR}
    rm -rf ${SYSTEMS_DIR}
fi

# Ensure /logs/systems is populated or abort process.
mkdir -p ${SYSTEMS_DIR}
cp /workspace/descriptions/systems/* ${SYSTEMS_DIR}/

# Populate /logs/calibration directory
mkdir -p ${CALIBRATION_DIR}
cp /workspace/workload_code/README.md ${CALIBRATION_DIR}/
cp /workspace/workload_code/run_calibration.sh ${CALIBRATION_DIR}/

# Populate /logs/code directory
mkdir -p ${CODE_DIR}
cp -r /workspace/workload_code/* ${CODE_DIR}/

# Populate /logs/measurements directory (No distibution between Offline and Server modes)
mkdir -p ${MEASUREMENTS_DIR}/Offline
cp /workspace/descriptions/measurements/* ${MEASUREMENTS_DIR}/Offline/
cp /workspace/workload_code/README.md ${MEASUREMENTS_DIR}/Offline/
cp /workspace/workload_code/user.conf ${MEASUREMENTS_DIR}/Offline/
cp /workspace/workload_code/run_calibration.sh ${MEASUREMENTS_DIR}/Offline/
cp /workspace/mlperf.conf ${MEASUREMENTS_DIR}/Offline/
cp -r ${MEASUREMENTS_DIR}/Offline ${MEASUREMENTS_DIR}/Server
