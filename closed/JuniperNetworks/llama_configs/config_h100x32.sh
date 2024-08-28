#!/bin/bash

export LOADGEN_CONFIG_DIR="build/loadgen-configs/DGX-H100_H100-SXM-80GBx32_TRT/llama2-70b-99.9/"
export NUM_GPUS=32
export SYSTEM_ID="DGX-H100_H100-SXM-80GBx32"
export GPU_RANK_MAP="0,1&2,3&4,5&6,7&0,1&2,3&4,5&6,7&0,1&2,3&4,5&6,7&0,1&2,3&4,5&6,7"
export DOCKER_ARGS="-v {Path}:{Path}"
export HISTFILE="{Path to Histfile}"
export HOST_NAME = "{HostName}"

export OMPI_MCA_orte_launch_agent="docker run --gpus=all  --rm -w /work \
        -e NCCL_IB_HCA=${NCCL_IB_HCA} -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} -e UCX_NET_DEVICES=${UCX_NET_DEVICES} \
        -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX} -e NCCL_DEBUG=${NCCL_DEBUG} -e NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS} \
        ${DOCKER_ARGS} \
        --cap-add SYS_ADMIN --cap-add SYS_TIME \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e HISTFILE=$HISTFILE \
        --shm-size=32gb \
        --ulimit memlock=-1 \
        -v /etc/timezone:/etc/timezone:ro -v /etc/localtime:/etc/localtime:ro \
	--security-opt apparmor=unconfined --security-opt seccomp=unconfined \
        --cpuset-cpus 0-223 \
        --user 1001 --net host --device /dev/fuse \
        --name $HOST_NAME -h $HOST_NAME \
        -e MLPERF_SCRATCH_PATH=$MLPERF_SCRATCH_PATH \
        -e HOST_HOSTNAME=$HOST_NAME \
        mlperf-inference:$CONT_NAME orted"

