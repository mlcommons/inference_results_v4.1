#!/bin/bash

SCENARIO=Offline ACCURACY=false bash run_mlperf.sh
SCENARIO=Server  ACCURACY=false bash run_mlperf.sh
SCENARIO=Offline ACCURACY=true  bash run_mlperf.sh
SCENARIO=Server  ACCURACY=true  bash run_mlperf.sh
