cd /work

echo $MLPERF_SCRATCH_PATH  # Make sure that the container has the MLPERF_SCRATCH_PATH set correctly
ls -al $MLPERF_SCRATCH_PATH  # Make sure that the container mounted the scratch space correctly

#!!!CAUTION, this will delete previous performance logs!!!
#make clean  # Make sure that the build/ directory isn't dirty
#!!!CAUTION, this will delete previous performance logs!!!

make link_dirs  # Link the build/ directory to the scratch space
ls -al build/ 

#Note: You should have already added custom systems at this point
# refer to READEME.md on adding systems using this script: 
#python3 -m scripts.custom_systems.add_custom_system

make build

#HPE fix for stable-diffusion convert to onnx not working
#pip install packaging
#sudo apt install -y --no-install-recommends apex
#git clone https://github.com/NVIDIA/apex
#cd apex
#pip install -v --disable-pip-version-check --no-cache-dir ./

#sudo apt install -y libjpeg-dev libpng-dev
#pip install torchvision 
#pip install "numpy<1.24,>=1.22" pip install "certifi==2022.12.7" "charset-normalizer==2.1.1" "filelock==3.8.2" "fsspec==2022.11.0" "idna==3.4" "Jinja2==3.1.2" "MarkupSafe==2.1.1" "requests==2.28.1" "tqdm==4.64.1" "typing-extensions==4.4.0" "urllib3==1.26.13"
#####

# The current build only supports SM89/SM90. If you want to try SM80 support, please go to Makefile.build and modify the "-a=90" flag from "build_trt_llm" target.
cd /work
make clone_trt_llm
make build_trt_llm
BUILD_TRTLLM=1 make build_harness

##then do make generate_engines ... commands


