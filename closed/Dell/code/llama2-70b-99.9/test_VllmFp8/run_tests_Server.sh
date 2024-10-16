#!/bin/bash

set -xeu

export SUBMISSION=${SUBMISSION:-1}
export SCENARIO="Server"
export TS_START_BENCHMARKS=${TS_START_BENCHMARKS:-`date +%m%d-%H%M%S`}
echo "TS_START_BENCHMARKS=${TS_START_BENCHMARKS}"

echo "Running $SCENARIO - Performance"
./test_VllmFp8_AsyncServer_perf.sh
echo "Running $SCENARIO - Accuracy"
./test_VllmFp8_AsyncServer_acc.sh
echo "Running $SCENARIO - Audit"
./test_VllmFp8_AsyncServer_audit.sh
echo "Done AsyncServer"
echo "TS_START_BENCHMARKS=${TS_START_BENCHMARKS}"