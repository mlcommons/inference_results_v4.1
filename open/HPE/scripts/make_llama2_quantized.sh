##These steps are from `./closed/NVIDIA/code/llama2-70b/tensorrt/README.md`
MLPERF_SCRATCH_PATH=/cstor/SHARED/datasets/MLPERF/inferencex/

cd ./closed/HPE/
git clone https://github.com/NVIDIA/TensorRT-LLM.git
# Using 2/6/2024 ToT
cd TensorRT-LLM && git checkout 0ab9d17a59c284d2de36889832fe9fc7c8697604

make -C docker build
# The default docker command will not mount extra directory. If necessary, copy the docker command and append
# -v <src_dir>:<dst:dir> to mount your own directory.

#copy model since not sure how to mount model_dir using make docker run...
mkdir ./temp_model
mkdir ./temp_model/Llama2
cp -r $MLPERF_SCRATCH_PATH/models/Llama2/Llama-2-70b-chat-hf/ ./temp_model/Llama2

make -C docker run LOCAL_USER=1 #DOCKER_ARGS="-v $MLPERF_SCRATCH_PATH:/model" #--mount type=bind,source=$MLPERF_SCRATCH_PATH

##################################################
# The following steps should be performed within TRTLLM container. Change -a=90 to your target architecture
git lfs install && git lfs pull
python3 ./scripts/build_wheel.py -a=90 --clean --install --trt_root /usr/local/tensorrt/

# Quantize the benchmark, On L40s, you might need TP4
python ./examples/quantization/quantize.py --dtype=float16  --output_dir=./temp_model/Llama2/fp8-quantized-ammo/llama2-70b-chat-hf-tp2pp1-fp8-02072024 --model_dir=./temp_model/Llama2/Llama-2-70b-chat-hf --qformat=fp8 --kv_cache_dtype=fp8 --tp_size 2
##################################################

#move temp_model back to scratch space
cp -r ./temp_model/Llama2/fp8-quantized-ammo/ $MLPERF_SCRATCH_PATH/models/Llama2/.
rm -rf ./temp_model

## Build and run the benchmarks
