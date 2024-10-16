# GPTJ readme

## Getting started

Please first download the data, model and preprocess the data folloing the steps below *within the mlperf container*. Note that if you have already downloaded the model and data prior to v4.1, you don't need to redo them. But you *need to re-run* the preprocess_data step for the updated calibration data.
```
BENCHMARKS=gptj make download_model
BENCHMARKS=gptj make download_data
BENCHMARKS=gptj make preprocess_data
```
Make sure after the 3 steps above, you have the model downloaded under `build/models/GPTJ-6B`, preprocessed data under `build/preprocessed_data/cnn_dailymail_tokenized_gptj/` and preprocessed calibration data (in HuggingFace Dataset format) under `build/preprocessed_data/gptj`.

### Backup model download method

If the download_data is not working due to MLCommon cloud error, you can use CK as an alternative apporach. We recommend to run it on a machine with sudo access.

```
pip install cmind
cm pull repo mlcommons@ck
cm run script --tags=get,ml-model,gptj,_pytorch,_rclone -j
```

The model will be rcloned to a local directory which looks likes: `/home/<username>/CM/repos/local/cache/04dedc0feede4f18/checkpoint`. Please move the model to `build/models/GPTJ-6B`.

### Note for non-SM90 systems
- If you are running on SM89 systems, change the build command to contain `-a="89-real"` to the `build_trt_llm` make target in `Makefile.build`. (If both 89 and 90 are needed, `-a="89-real;90"`)
- If you are running on SM80 systems, change the build command to contain -a="80". Note that SM80 doesn't have FP8, and is not officially supported.

## Build and quantization preparation for the GPTJ on datacenter systems

GPTJ on hopper submission use FP8 GPT-J model. Here is the steps to obtain the model.
```
# Enter the MLPerf container
make prebuild
# If you have not built TRTLLM yet, or the TRTLLM is outdated, do:
rm -rf build/TRTLLM && make clone_trt_llm && make build_trt_llm
# If you have built the latest TRTLLM, but just have not installed it yet, do:
SKIP_TRTLLM_BUILD=1 make build_trt_llm

# Quantize the benchmark
python build/TRTLLM/examples/quantization/quantize.py --model_dir build/models/GPTJ-6B/checkpoint-final --dtype float16 --qformat fp8 --kv_cache_dtype fp8 --output_dir build/models/GPTJ-6B/fp8-quantized-ammo/GPTJ-FP8-quantized --calib_size 1024 --tp_size 1 --calib_dataset build/preprocessed_data/gptj/mlperf_gptj_openorca_calibration_1k/
```

## Build and quantization preparation for the GPTJ on Orin

On Orin, GPTJ is using int4 weight-only awq quantization, in order to obtain this quantized model, we need to use TRTLLM to quantize the model from downloaded public checkpoint.
```
# Enter the MLPerf container
make prebuild
# If you have not built TRTLLM yet, or the TRTLLM is outdated, do:
rm -rf build/TRTLLM && make clone_trt_llm && make build_trt_llm
# If you have built the latest TRTLLM, but just have not installed it yet, do:
SKIP_TRTLLM_BUILD=1 make build_trt_llm

# Quantize the benchmark
python build/TRTLLM/examples/quantization/quantize.py --model_dir ${MLPERF_SCRATCH_PATH}/models/GPTJ-6B/checkpo
int-final/ --dtype float16 --qformat int4_awq --output_dir ${MLPERF_SCRATCH_PATH}/models/GPTJ-6B/orin-w4a16-awq --calib_size 512
```

## Build and run the benchmarks on datacenter systems

Please follow the steps below:

```
# make sure you are in mlperf's container
make prebuild
# if not, make sure you already built TRTLLM as well as mlperf harnesses needed for GPTJ run.
make build
# If the latest TRTLLM is already built in the steps above, you can expedite the build. You don't need to run make build if loadgen, TRTLLM, and harnesses are already built on the latest commit.
SKIP_TRTLLM_BUILD=1 make build

# For Datacenter submission Before generating the engines, please point fp8_quant_model_path in code/gptj/tensorrt/builder.py to your quantized model path.

make generate_engines RUN_ARGS="--benchmarks=gptj --scenarios=Offline --config_ver=high_accuracy --test_mode=AccuracyOnly"
make run_harness RUN_ARGS="--benchmarks=gptj --scenarios=Offline --config_ver=high_accuracy --test_mode=AccuracyOnly"
```

On Hopper machine, You should expect to get the following results:
```
  gptj-99.9:
     accuracy: [PASSED] ROUGE1: 43.102 (Threshold=42.944) | [PASSED] ROUGE2: 20.113 (Threshold=20.103) | [PASSED] ROUGEL: 29.975 (Threshold=29.958) | [PASSED] GEN_LEN: 4114386.000 (Threshold=3615190.200)
```

## Build and run the benchmarks on Orin
Please follow the steps below:

```
# make sure you are in mlperf's container
make prebuild
# if not, make sure you already built TRTLLM as well as mlperf harnesses needed for GPTJ run.
make build
# If the latest TRTLLM is already built in the steps above, you can expedite the build by
SKIP_TRTLLM_BUILD=1 make build

# For Orin submission, make sure the variable int4_quant_model_path in code/gptj/tensorrt/builder.py points to your quantized int4 model path.

make generate_engines RUN_ARGS="--benchmarks=gptj --scenarios=Offline --config_ver=default --test_mode=AccuracyOnly"
make run_harness RUN_ARGS="--benchmarks=gptj --scenarios=Offline --config_ver=default --test_mode=AccuracyOnly"
```

On Orin, You should expect to get the following results:
```
  gptj-99.9:
     accuracy: [PASSED] ROUGE1: 43.068 (Threshold=42.944) | [PASSED] ROUGE2: 20.129 (Threshold=20.103) | [PASSED] ROUGEL: 30.022 (Threshold=29.958) | [PASSED] GEN_LEN: 4095514.000 (Threshold=3615190.200)
```
