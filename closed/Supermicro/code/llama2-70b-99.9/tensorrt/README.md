# Llama2 readme

## Getting started

Please first download the data, model and preprocess the data folloing the steps below *within the mlperf container*. Note that if you have already downloaded the model and data prior to v4.1, you don't need to redo them. But you *need to re-run* the preprocess_data step for the updated calibration data.
```
# Visit https://docs.google.com/forms/d/e/1FAIpQLSc_8VIvRmXM3I8KQaYnKf7gy27Z63BBoI_I1u02f4lw6rBp3g/viewform to sign the agreement,
# which gets you access to the dataset
# Unzip the llama dataset pickle file
gzip -d <llama_dataset>.pkl.gz
# Please place the llama2 models (Llama-2-70b-chat-hf) under build/models/Llama2 and datasets (open_orca) under build/preprocessed_data
# This step will create the necessary npy file for harness, and dataset parquet for calibration
BENCHMARKS=llama2-70b make preprocess_data
```
Make sure after the 3 steps above, you have the model downloaded under `build/models/Llama2`, and preprocessed data and calibration data under `build/preprocessed_data/open_orca/`.

### Note for non-SM90 systems
- If you are running on SM89 systems, change the build command to contain `-a="89-real"` to the `build_trt_llm` make target in `Makefile.build`. (If both 89 and 90 are needed, `-a="89-real;90"`)
- If you are running on SM80 systems, change the build command to contain -a="80". Note that SM80 doesn't have FP8, and is not officially supported.

## Build and quantization preparation for the Llama2

```
# Enter the MLPerf container
make prebuild
# If you have not built TRTLLM yet, or the TRTLLM is outdated, do:
rm -rf build/TRTLLM && make clone_trt_llm && make build_trt_llm
# If you have built the latest TRTLLM, but just have not installed it yet, do:
SKIP_TRTLLM_BUILD=1 make build_trt_llm

# Quantize the benchmark. The calibration dataset file is generated in the preprocess_data step above
# On L40s, you need --tp_size=4. on H200 you just need --tp_size 1
python build/TRTLLM/examples/quantization/quantize.py --dtype float16 --qformat fp8 --kv_cache_dtype fp8 --output_dir=build/models/Llama2/fp8-quantized-ammo/llama2-70b-chat-hf-tp1pp1-fp8 --model_dir=build/models/Llama2/Llama-2-70b-chat-hf --calib_size 1024  --tp_size 1 --calib_dataset build/preprocessed_data/open_orca/mlperf_llama2_openorca_calibration_1k/
```

## Build and run the benchmarks

Please follow the steps below in MLPerf container:
```
# make sure you are in mlperf's container
make prebuild
# if not, make sure you already built TRTLLM as well as mlperf harnesses needed for GPTJ run.
make build
# If the latest TRTLLM is already built in the steps above, you can expedite the build. You don't need to run make build if loadgen, TRTLLM, and harnesses are already built on the latest commit.
SKIP_TRTLLM_BUILD=1 make build

# Please update configs/llama2-70b to include your custom machine config before building the engine
make generate_engines RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline --config_ver=high_accuracy"
make run_harness RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline --config_ver=high_accuracy --test_mode=AccuracyOnly"
```

For a general rule of thumb, GPUs with:
- ~40GB of VMEM needs tensor parallelism of 4
- ~80GB of VMEM needs tensor parallelism of 2
- > 90GB of VMEM can run tensor parallelism of 1.

You should expect to get the following results (the detailed number might be different):
```
   accuracy: [PASSED] ROUGE1: 44.495 (Threshold=43.836) | [PASSED] ROUGE2: 22.089 (Threshold=21.689) | [PASSED] ROUGEL: 28.694 (Threshold=28.222) | [PASSED] TOKENS_PER_SAMPLE: 293.100 (Threshold=263.970)
```
