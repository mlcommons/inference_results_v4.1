## Preprocess all data
#make preprocess_data

## Preprocess specific data
#make preprocess_data BENCHMARKS="resnet50 retinanet bert rnnt 3d-unet gptj6b" dlrm_v2 gpt175b
make preprocess_data BENCHMARKS="resnet50"
make preprocess_data BENCHMARKS="retinanet"
make preprocess_data BENCHMARKS="bert"
#make preprocess_data BENCHMARKS="rnnt" #removed in mlperf v4.1
make preprocess_data BENCHMARKS="3d-unet"
#make preprocess_data BENCHMARKS="dlrm_v2" #skipped
make preprocess_data BENCHMARKS="stable-diffusion-xl"
make preprocess_data BENCHMARKS="gptj"
echo "GPT-J needs quantization, refer to ./closed/NVIDIA/code/gptj/tenorrt/README.md and ./closed/HPE/scripts/make_gptj_quantized.sh for instructions"

#Special steps for Llama2, see `./closed/NVIDIA/code/llama2-70b/tenorrt/README.md`
gzip -d $MLPERF_SCRATCH_PATH/preprocessed_data/open_orca/*.pkl.gz
make preprocess_data BENCHMARKS="llama2-70b"
echo "Llama2-70b needs quantization, refer to ./closed/NVIDIA/code/llama2-70b/tenorrt/README.md and ./closed/HPE/scripts/make_llama2_quantized.sh for instructions"

#Special steps for Mixtral in progress...
make preprocess_data BENCHMARKS="mixtral-8x7b"
echo "Mixtral-8x7b needs quantization, refer to ./closed/NVIDIA/code/mixtral-8x7b/tenorrt/README.md and ./closed/HPE/scripts/make_mixtral_quantized.sh for instructions"

