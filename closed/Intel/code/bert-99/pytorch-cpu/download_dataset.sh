#!/bin/bash

export DATA_DIR=/data
wget 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json' -O "${DATA_DIR}/dev-v1.1.json"
