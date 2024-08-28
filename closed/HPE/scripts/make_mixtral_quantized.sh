##These steps are from `./closed/NVIDIA/code/mixtral-8x70b/tensorrt/README.md`

######################
#These steps needed for NVIDIA partner-build 4.1.2
######################

#make clone_trt_llm && make build_trt_llm #if not completed already
# Change the TP size accordingly. 1 should be enough for >=80 GB system
#cd build/TRTLLM/examples/mixtral
#TP1
#python ../quantization/quantize.py --model_dir /work/build/models/Mixtral/Mixtral-8x7B-Instruct-v0.1 --dtype float16 --qformat fp8 --kv_cache_dtype fp8 --output_dir /work/build/models/Mixtral/fp8-quantized-ammo/mixtral-8x7b-instruct-v0.1-tp1pp1-fp8 --calib_size 1024 --tp_size 1

#TP2
#python ../quantization/quantize.py --model_dir /work/build/models/Mixtral/Mixtral-8x7B-Instruct-v0.1 --dtype float16 --qformat fp8 --kv_cache_dtype fp8 --output_dir /work/build/models/Mixtral/fp8-quantized-ammo/mixtral-8x7b-instruct-v0.1-tp2pp1-fp8 --calib_size 1024 --tp_size 2


######################
#NVIDIA partner-build 4.1.3 copies the quantized model directly from container instead of above steps
######################
mixtral_dir=build/models/Mixtral/fp8-quantized-modelopt
mkdir $mixtral_dir
tar -xvf /opt/fp8-quantized-modelopt/mixtral-8x7b-instruct-v0.1-tp1pp1-fp8.tar.gz $mixtral_dir/