#!/bin/bash
BASE_IMAGE=rocm/pytorch:rocm6.1.2_ubuntu20.04_py3.9_pytorch_staging
VLLM_REV=b7d5e159fea83a75be94c6d294fd39e4bbed5006 # MLPerf-4.1
HIPBLASLT_BRANCH=8b71e7a8d26ba95774fdc372883ee0be57af3d28
FA_BRANCH=23a2b1c2f21de2289db83de7d42e125586368e66 # ck_tile - FA 2.5.9

git clone https://github.com/ROCm/vllm
pushd vllm
git checkout main
git pull
git checkout ${VLLM_REV}

docker build --build-arg BASE_IMAGE=${BASE_IMAGE} --build-arg HIPBLASLT_BRANCH=${HIPBLASLT_BRANCH} --build-arg FA_BRANCH=${FA_BRANCH} -f Dockerfile.rocm -t vllm_dev:${VLLM_REV} .

popd

docker build --build-arg BASE_IMAGE=vllm_dev:${VLLM_REV} -f Dockerfile.llama2 -t mlperf/llama_inference:20240724b ..
