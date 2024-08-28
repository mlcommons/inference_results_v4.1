##These steps are from `./closed/NVIDIA/code/gptj/tensorrt/README.md`
MLPERF_SCRATCH_PATH=/cstor/SHARED/datasets/MLPERF/inferencex/

cd ./closed/HPE/

git clone https://github.com/NVIDIA/TensorRT-LLM.git
# Using 2/6/2024 ToT
cd TensorRT-LLM && git checkout 0ab9d17a59c284d2de36889832fe9fc7c8697604
cp -r ../code/gptj/tensorrt . #this is needed for last step (onnx_tune.py)


make -C docker build
# The default docker command will not mount extra directory. If necessary, copy the docker command and append
# -v <src_dir>:<dst:dir> to mount your own directory.

#copy model since not sure how to mount model_dir using make docker run...
mkdir ./temp_model
cp -r $MLPERF_SCRATCH_PATH/models/GPTJ-6B/ temp_model/

make -C docker run LOCAL_USER=1 #DOCKER_ARGS="-v $MLPERF_SCRATCH_PATH:/model" #--mount type=bind,source=$MLPERF_SCRATCH_PATH

##################################################
# The following steps should be performed within TRTLLM container. Change -a=90 to your target architecture
git lfs install && git lfs pull
python3 ./scripts/build_wheel.py -a=90 --clean --install --trt_root /usr/local/tensorrt/

# Quantize the benchmark
python ./examples/quantization/quantize.py --dtype=float16  --output_dir=temp_model/GPTJ-6B/fp8-quantized-ammo/GPTJ-FP8-quantized-02072024/ --model_dir=temp_model/GPTJ-6B/checkpoint-final --qformat=fp8 --kv_cache_dtype=fp8

# Further tune the quantization in-place
python ./tensorrt/onnx_tune.py --fp8-scalers-path=temp_model/GPTJ-6B/fp8-quantized-ammo/GPTJ-FP8-quantized-02072024/rank0.safetensors --scaler 1.005 --index 15
##################################################

# exit container and move temp_model back to scratch space
cp -r temp_model/GPTJ-6B/fp8-quantized-ammo/ $MLPERF_SCRATCH_PATH/models/GPTJ-6B/
rm -rf ./temp_model

