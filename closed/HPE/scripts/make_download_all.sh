## Downloads all models
#make download_model

## Download specific models
#make download_model BENCHMARKS="resnet50 retinanet bert rnnt 3d-unet gptj" # dlrm_v2 gpt175b
make download_model BENCHMARKS="resnet50"
make download_model BENCHMARKS="retinanet"
make download_model BENCHMARKS="bert"
#make download_model BENCHMARKS="rnnt"  #removed in mlperf v4.1
make download_model BENCHMARKS="3d-unet"
make download_model BENCHMARKS="gptj"
make download_model BENCHMARKS="stable-diffusion-xl"
echo "if downloading fails for GPT-J, DLRMv2, Stable-diffusion models then refer to this instruction to download using rclone: https://groups.google.com/a/mlcommons.org/g/inference/c/bUplKnzCCoM/m/zaqf3HUgAAAJ"
echo "Llama2-70b model must be manually downloaded. Use this link to for download steps using rclone on a remote system: https://groups.google.com/a/mlcommons.org/g/inference/c/bUplKnzCCoM/m/Z_LWKPuWAAAJ"

make download_model BENCHMARKS="mixtral-8x7b" #new for v4.1
echo "Mixtral-8x7B requires git clone from huggingface.co"
    mkdir build/models/Mixtral
    cd build/models/Mixtral && git clone https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1
    # Checkout to the older version to make sure the tokenizer is aligned
    cd Mixtral-8x7B-Instruct-v0.1 && git checkout 1e637f2d7cb0a9d6fb1922f305cb784995190a83

## Downloads all data
#make download_data

## Downloads specific data
#make download_data BENCHMARKS="resnet50 retinanet bert rnnt 3d-unet gptj" # dlrm_v2 gpt175b
make download_data BENCHMARKS="resnet50"
make download_data BENCHMARKS="retinanet"
make download_data BENCHMARKS="bert"
#make download_data BENCHMARKS="rnnt" #removed in mlperf v4.1
make download_data BENCHMARKS="3d-unet"
make download_data BENCHMARKS="gptj"
make download_data BENCHMARKS="stable-diffusion-xl"
echo "Llama2-70b dataset must be manually downloaded. Use this link to for download steps using rclone on a remote system: https://groups.google.com/a/mlcommons.org/g/inference/c/bUplKnzCCoM/m/Z_LWKPuWAAAJ"
make download_data BENCHMARKS="mixtral-8x7b" #new for v4.1
