# MoE Mixtral-8x7B readme

## Getting started

Please first download the data, model and preprocess the data following the steps below within the mlperf container (make sure build/data, build/preprocessed_data, build/models exist)
```
BENCHMARKS=mixtral-8x7b make download_data
BENCHMARKS=mixtral-8x7b make preprocess_data
cd build/models/Mixtral && git clone https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1
# Checkout to the older version to make sure the tokenizer is aligned
cd Mixtral-8x7B-Instruct-v0.1 && git checkout 1e637f2d7cb0a9d6fb1922f305cb784995190a83
```
Make sure after the steps above, you have the model downloaded under `build/models/Mixtral`, and preprocessed data and calibration data under `build/preprocessed_data/moe/`.

### Note for non-SM90 systems
- If you are running on SM89 systems, change the build command to contain `-a="89-real"` to the `build_trt_llm` make target in `Makefile.build`. (If both 89 and 90 are needed, `-a="89-real;90"`)
- If you are running on SM80 systems, change the build command to contain -a="80". Note that SM80 doesn't have FP8, and is not officially supported.

## Build and quantization preparation for the Mixtral

[Note] the FP8 quantized model (tensor parallelism = 1) is provided in the release container (`/opt/fp8-quantized-modelopt/mixtral-8x7b-instruct-v0.1-tp1pp1-fp8.tar.gz`). Please untar it and place the dir to `build/models/Mixtral/fp8-quantized-modelopt/mixtral-8x7b-instruct-v0.1-tp1pp1-fp8`.

## Build and run the benchmarks

Please follow the steps below in MLPerf container:
```
# make sure you are in mlperf's container
make prebuild
# if not, make sure you already built TRTLLM as well as mlperf harnesses needed for GPTJ run.
make build
# If the latest TRTLLM is already built in the steps above, you can expedite the build. You don't need to run make build if loadgen, TRTLLM, and harnesses are already built on the latest commit.
SKIP_TRTLLM_BUILD=1 make build

make generate_engines RUN_ARGS="--benchmarks=mixtral --scenarios=Offline"
make run_harness RUN_ARGS="--benchmarks=mixtral --scenarios=Offline --test_mode=AccuracyOnly"
```

For a general rule of thumb, GPUs with:
- ~40GB of VMEM needs tensor parallelism of 2
- >=80GB of VMEM needs tensor parallelism of 1

You should expect to get the following results (the detailed board and numbers are different):
```
 H200-SXM-141GBx1_TRT-custom_k_99_MaxP-Offline:
   mixtral-8x7b:
     accuracy: [PASSED] ROUGE1: 45.356 (Threshold=45.036) | [PASSED] ROUGE2: 23.249 (Threshold=23.050) | [PASSED] ROUGEL: 30.316 (Threshold=30.058) | [PASSED] TOKENS_PER_SAMPLE: 145.900 (Valid Range=[131.310,  160.490]) | [PASSED] gsm8k_accuracy: 73.100 (Threshold=73.042) | [PASSED] mbxp_accuracy: 59.880 (Threshold=59.519)
```

# For Internal MLPerf developer (not needed for external partners)

## Build and run the benchmarks in standalone TRTLLM mode (through infer_trtllm.py)
Build the engine for Mixtral first
```
make generate_engines RUN_ARGS="--benchmarks=mixtral --scenarios=Offline"
```

Run the infer_trtllm.py
```
python3 /work/code/mixtral-8x7b/tensorrt/infer_trtllm.py --tllm_engine_dir /work/build/engines/H200-SXM-141GBx8/mixtral-8x7b/Offline/bs806-custom_k_99_MaxP-tp1-pp1 --bs=128 --samples=15000
```

Output will be saved in `trtllm_mixtral_8x7b_{len(output_tokens)}_BS{args.bs}_greedy.pkl`

## Build and run the benchmarks in reference implementation mode (through infer_hf.py)
Within the container
```
python3 infer_hf.py --bs=64 --samples=15000
```
Note that you might need to lower the BS to avoid OOM. Output will be saved in `mixtral_8x7b_{len(output_tokens)}_BS{args.bs}_greedy_hf_ref_fp16.pkl`


## Run the accuracy check in standalone mode (for infer_*.py)
From outside the container, Build and run the evaluation container.
```
cd code/mixtral-8x7b/tensorrt/
docker build -f ./Dockerfile.accuracy --tag docker-codegen-eval .
docker run -it --rm --net=host --runtime=nvidia --ipc=host -v $PWD:$PWD -w $PWD docker-codegen-eval

# For TRTLLM results (substitute the pickle path)
python3 accuracy_moe.py --results_path=trtllm_mixtral_8x7b_15000_BS128_greedy.pkl --result_key=nv_tllm_ref_output --length_key=nv_tllm_tok_ref_output_length
```
