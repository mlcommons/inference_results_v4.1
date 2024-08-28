#!/bin/bash

export LOG_DIR="${LOG_DIR:-${TMP_DIR}}"
source ./run_common.sh

common_opt="--config ./mlperf.conf"

OUTPUT_DIR=${LOG_DIR}
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

set -x # echo the next command

profiling=0
if [ $profiling == 1 ]; then
    EXTRA_OPS="$EXTRA_OPS --enable-profiling=True"
fi

export number_cores=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
python ./user_config.py

## multi-instance
python -u python/runner.py --profile $profile $common_opt --model $model --model-path $model_path \
                           --dataset $dataset --dataset-path ${DATA_DIR} --output $OUTPUT_DIR $EXTRA_OPS $@
