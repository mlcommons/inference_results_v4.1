# Llama2 readme

## Getting started

Please first download the data, model and preprocess the data folloing the steps below
```
# Visit https://docs.google.com/forms/d/e/1FAIpQLSc_8VIvRmXM3I8KQaYnKf7gy27Z63BBoI_I1u02f4lw6rBp3g/viewform to sign the agreement,
# which gets you access to the dataset
# Unzip the llama dataset pickle file
gzip -d <llama_dataset>.pkl.gz
# Please place the llama2 models (Llama-2-70b-chat-hf) under build/models/Llama2 and datasets (open_orca) under build/preprocessed_data
# This step will create the necessary npy file for harness, and dataset parquet for calibration
BENCHMARKS=llama2 make preprocess_data
```
Make sure after the 3 steps above, you have the model downloaded under `build/models/Llama2`, and preprocessed data and calibration data under `build/preprocessed_data/open_orca/`.

## Build and preparation for the Llama2

```
# Enter the MLPerf container
make prebuild
# Make sure TRTLLM is built, add SKIP_TRTLLM_BUILD=1 if it's alredy built beforehand.
make build_trt_llm

```

NVIDIA has multiple optimized Llama2 submissions on both data center and edge for v4.1. The models we optimized are not publicly shared because of concerns of license. But they are available through email request to inference-chairs@mlcommons.org

## Build and run DepthPruned benchmarks

The DepthPruned Llama2 is a model we created by pruning out: 1. layers of the original Llama2 model; and 2. dimensions of the MLP layers. We then use SFT (supervised fine tuning) to recover accuracy.

On the data center H200 GPU platforms, we quantize the pruned model to FP8. And on the edge Orin GPU platforms, we quantize the pruned model to INT4 weight only using AWQ. The pruned models we release are already quantized.

Please follow the steps below in MLPerf container to build and run the DepthPruned benchmarks:
```
# The current build only supports SM89/SM90. If you want to try SM80 support, please go to Makefile.build and modify the "-a=90" flag from "build_trt_llm" target.
make build
# Please change model_path and quant_model_path field in the configs/llama2-70b/Offline to point to your quantized model directory
# Please update configs/llama2-70b to include your custom machine config before building the engine
make generate_engines RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline --config_ver=high_accuracy --model_opt=DepthPruned"
make run_harness RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline --config_ver=high_accuracy --test_mode=AccuracyOnly --model_opt=DepthPruned"
```

You should expect to get the results similar to the following:
```
   llama2-70b-99:
     accuracy: [] ROUGE1: 42.978 (Threshold=43.987) | [[PASSED] ] ROUGE2: 24.337 (Threshold=21.815) | [[PASSED] ] ROUGEL: 31.404 (Threshold=28.330) | [] TOKENS_PER_SAMPLE: 164.600 (Threshold=265.005)

```

## Build and run Sparse benchmarks

The Sparse Llama2 is a model we created by sparsify all the GEMMs. We then use SFT (supervised fine tuning) to recover accuracy.

We quantize the pruned model to FP8. The pruned models we release are already quantized.

Please follow the steps below in MLPerf container to build and run the Sparse benchmarks:
```
# The current build only supports SM89/SM90. If you want to try SM80 support, please go to Makefile.build and modify the "-a=90" flag from "build_trt_llm" target.
make build
# Please change model_path and quant_model_path field in the configs/llama2-70b/Offline to point to your quantized model directory
# Please update configs/llama2-70b to include your custom machine config before building the engine
make generate_engines RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline --config_ver=high_accuracy --model_opt=Sparse"
make run_harness RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline --config_ver=high_accuracy --test_mode=AccuracyOnly --model_opt=Sparse"
```

You should expect to get the results similar to the following:
```
   llama2-70b-99:
     accuracy: [[PASSED] ] ROUGE1: 44.460 (Threshold=43.987) | [[PASSED] ] ROUGE2: 22.453 (Threshold=21.815) | [[PASSED] ] ROUGEL: 29.949 (Threshold=28.330) | [] TOKENS_PER_SAMPLE: 234.400 (Threshold=265.005)
```
