#!/bin/bash

#export DATA_DIR=/root/mlperf_data
export DATA_DIR=/data
cd ${DATA_DIR}
git clone https://github.com/neheller/kits19
cd kits19
python3 -m starter_code.get_imaging
