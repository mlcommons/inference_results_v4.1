#!/bin/bash

export TMP_DIR="/tmp/submission_original"
export TRUNK_DIR="/tmp/submission_truncated"
export VENDOR=OEM

source default.conf
export LOG_DIR="${LOG_DIR:-/logs}"
export DEL_FILES="${DEL_FILES:-${DEL_FILES_DEFAULT}}"

# Creating temporary directories for submission check pre-processing
mkdir -p ${TMP_DIR}/closed/${VENDOR}
cp -r ${LOG_DIR}/* ${TMP_DIR}/closed/${VENDOR}/

# Removing troublesome files
for FILE in ${DEL_FILES}; do
    rm ${TMP_DIR}/${FILE}
done

# Begining MLPerf Scripts
cd /workspace/inference
echo "Truncating accuracy logs"
python tools/submission/truncate_accuracy_log.py --input ${TMP_DIR} --submitter ${VENDOR} --output ${TRUNK_DIR}
echo "Begining submission check on truncated results"
python3 tools/submission/submission_checker.py --input ${TRUNK_DIR} --submitter=${VENDOR} --version=v4.1

# Clearing the temporary directores
rm -rf ${TMP_DIR}
rm -rf ${TRUNK_DIR}
