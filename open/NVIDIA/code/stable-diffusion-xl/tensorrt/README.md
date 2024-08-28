# Stable Diffusion XL Benchmark


## Getting started

Please first download the data, closed division model and preprocess the data folloing the steps below
```
BENCHMARKS=stable-diffusion-xl make download_model
BENCHMARKS=stable-diffusion-xl make download_data
BENCHMARKS=stable-diffusion-xl make preprocess_data
```
The closed division PyTorch model is downloaded from the [Hugging Face snapshot](https://cloud.mlcommons.org/index.php/s/DjnCSGyNBkWA4Ro) provided by the MLCommon. The Pytorch model is subsequently processed into 4 onnx models. Make sure after the 3 steps above, you have the closed division models downloaded under `build/models/SDXL/onnx_models`, and preprocessed data under `build/preprocessed_data/coco2014-tokenized-sdxl/`.

NVIDIA has one open SDXL submission for v4.1 which is based on Latent Consistency Model (LCM).

All the model quantizations and other preparations are the same as in NVIDIA's closed division. Please refer the README doc over there.

## Build and run the LCM benchmarks

Please follow the steps below in MLPerf container:

```
make build

# Please update configs/stable-diffusion-xl to include your custom machine config before building the engine
make generate_engines RUN_ARGS="--benchmarks=stable-diffusion-xl --scenarios=Offline --model_opt=LCM"

make run_harness RUN_ARGS="--benchmarks=stable-diffusion-xl --scenarios=Offline --test_mode=AccuracyOnly --model_opt=LCM"
```

You should expect to get the following results:
```
   stable-diffusion-xl:
     accuracy: [] CLIP_SCORE: 31.280 (Valid Range=[31.686,  31.813]) | [] FID_SCORE: 28.428 (Valid Range=[23.011,  23.950])

```
