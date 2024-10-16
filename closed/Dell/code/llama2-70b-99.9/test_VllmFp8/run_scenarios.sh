#!/bin/bash

set -xeu

export SUBMISSION=${SUBMISSION:-1}
# export SCENARIO="Server"
export TS_START_BENCHMARKS=${TS_START_BENCHMARKS:-`date +%m%d-%H%M%S`}
echo "TS_START_BENCHMARKS=${TS_START_BENCHMARKS}"

echo "Running Offline"
./run_VllmFp8_Offline.sh
echo "Done Offline"

echo "Running Server"
./run_VllmFp8_AsyncServer.sh
echo "Done Server"

echo "Done Benchmarks"
echo "TS_START_BENCHMARKS=${TS_START_BENCHMARKS}"