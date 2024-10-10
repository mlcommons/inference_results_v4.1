# AMD MI300X Llama2-70b
## Setup
### Model Preparation
Download the Llama2-70b weights using the instructions in https://github.com/mlcommons/inference/tree/master/language/llama2-70b#get-model. Download the model to a path specified in environment variable `$LAB_MODEL`.

### Dataset Preparation
Download the preprocessed dataset files using the instructions in https://github.com/mlcommons/inference/tree/master/language/llama2-70b#preprocessed to a directory specified by `$LAB_DATASET`.

### AMD MLPerf Inference Docker Container Setup
Build the Docker image and launch a container using the commands below. The commands should be run from the top level of this repo. Set the environment variable `$LAB_HIST` with the directory where benchmark outputs will be stored.
``` bash
cd docker

# Build the image `mlperf/llama_inference:latest`
./build_llama2.sh

# Launch a docker container
docker run -it --ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
    --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -v ${LAB_MODEL}:/data/llm/llama2-70b-chat \
    -v ${LAB_DATASET}:/data/open_orca \
    -v ${LAB_HIST}:/lab-hist \
    -e LAB_CLOG=/lab-hist/mlperf-results \
    mlperf/llama_inference:latest
```

### Quantization Preparation
Quantize the model by running the instructions below in the inference container
``` bash
model_dir=/data/llm/llama2-70b-chat
output_dir=/data/llm/llama2-70b-chat/quantized/quark_share/modelzoo/llama2_70b_wfp8_afp8_ofp8_nomerge/json-safetensors/llama.safetensors
calib_dataset=/data/open_orca/open_orca_gpt4_tokenized_llama.calibration_1000.pkl.gz

cd /lab-mlperf-inference/code/llama2-70b-99.9/tools/quark-0.1.0+a9827f5-mlperf/examples/torch/language_modeling/
python3 quantize_quark.py --model_dir $model_dir \
    --output_dir $output_dir \
    --quant_scheme w_fp8_a_fp8_o_fp8 \
    --dataset $calib_dataset \
    --num_calib_data 1000 \
    --model_export vllm_adopted_safetensors \
    --no_weight_matrix_merge
```

KV cache scales for the quantized model weights are used and were downloaded from https://github.com/vllm-project/vllm/blob/38c4b7e863570a045308af814c72f4504297222e/tests/fp8_kv/llama2-70b-fp8-kv/kv_cache_scales.json.


## Reproduce Results
To generate results for the full submission, running the command below in an inference container. Logs can be found in `/lab-hist/mlperf-results/$datetime1/$datetime2`.
``` bash
cd /lab-mlperf-inference/code/llama2-70b-99.9/test_VllmFp8
bash ./run_scenarios.sh
```

To generate results for the Offline scenario only, run the command below in an inference container. Logs can be found in `/lab-hist/mlperf-results/$datetime1/$datetime2/Offline`.
``` bash
cd /lab-mlperf-inference/code/llama2-70b-99.9/test_VllmFp8
bash ./run_tests_Offline.sh
```

To generate results for the Server scenario only, run the command below in an inference container. Logs can be found in `/lab-hist/mlperf-results/$datetime1/$datetime2/Server`.
``` bash
cd /lab-mlperf-inference/code/llama2-70b-99.9/test_VllmFp8
bash ./run_tests_Server.sh
```
