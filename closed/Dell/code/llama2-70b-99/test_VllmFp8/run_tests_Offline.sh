#!/bin/bash

set -xeu

export SUBMISSION=${SUBMISSION:-1}
export SCENARIO="Offline"
export TS_START_BENCHMARKS=${TS_START_BENCHMARKS:-`date +%m%d-%H%M%S`}
echo "TS_START_BENCHMARKS=${TS_START_BENCHMARKS}"

echo "Running $SCENARIO - Performance"
./test_VllmFp8_Offline_perf.sh
echo "Running $SCENARIO - Accuracy"
./test_VllmFp8_Offline_acc.sh
echo "Running $SCENARIO - Audit"
./test_VllmFp8_Offline_audit.sh
echo "Done"
echo "TS_START_BENCHMARKS=${TS_START_BENCHMARKS}"