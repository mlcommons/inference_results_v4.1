# Instructions on how to run SDXL on TPU


## Setup
### Create 1 TPU v5e-4 VM.
Follow these [instructions](https://cloud.google.com/tpu/docs/v5e-inference#tpu-vm) to create TPUv5e-4 VMs.
Use `v2-alpha-tpuv5-lite` as `version`/`runtime-vesrion`

### Setup Code & Dependencies
ssh into the VM

git clone https://github.com/mlcommons/inference.git

cd `inference/text_to_image/` and pip install -r requirements.txt

### Install JAX dependencies
pip install jax[tpu]==0.4.30 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

pip install flax==0.8.5


From the Readme here, install and setup loadgen

Copy the `sdlx_tpu` found in the submission package, into the cloned repo `inference/text_to_image/` directory.

Then make sure you're in the directory  `inference/text_to_image/sdxl_tpu`.
`pip install -r requirements.txt`

set ROOT_DIRECTORY as `inference/text_to_image/sdxl_tpu`.

set OUTPUT_PATH to where the output logs can be saved.

-------

## Offline Scenario 

### Performance 
```python
python3 main.py --config=$ROOT_DIRECTORY/configs/base_xl.yml --latents=$ROOT_DIRECTORY/coco2014/latents/latents.npy --threads=48 --scenario=Offline --threshold-time=8 --threshold-queue-length=4 --max-batchsize=4 --output={OUTPUT_PATH}

```


### Accuracy
```python
python3 main.py --config=$ROOT_DIRECTORY/configs/base_xl.yml --latents=$ROOT_DIRECTORY/coco2014/latents/latents.npy --threads=48 --scenario=Offline  --threshold-time=8 --threshold-queue-length=4 --output={OUTPUT_PATH} --accuracy
```



-----

## Server Scenario 


## Performance
```python
python3 main.py --config=$ROOT_DIRECTORY/configs/base_xl.yml  --latents=$ROOT_DIRECTORY/coco2014/latents/latents.npy --threads=48 --threshold-time=8 --threshold-queue-length=4 --scenario=Server --output={OUTPUT_PATH}

```


## Accuracy 
```python
python3 main.py --config=$ROOT_DIRECTORY/configs/base_xl.yml --latents=$ROOT_DIRECTORY/coco2014/latents/latents.npy --threads=48 --threshold-time=8 --threshold-queue-length=4  --scenario=Server --output={OUTPUT_PATH} --accuracy

```


----




