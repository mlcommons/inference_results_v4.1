#!/bin/bash

MODEL_DIR=/model
cm run script --tags=get,ml-model,dlrm,_pytorch,_weight_sharded,_rclone -j --to=${MODEL_DIR}
