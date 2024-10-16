# MLPerf Inference - ResNet50 - UntetherAI

This implementation runs the MLPerf Inference ResNet50 workload on UntetherAI
speedAI240 accelerator boards using KRAI Inference Library Technology (KILT).


# Detailed instructions

## Install the `axs` workflow automation technology

### Kernel
```
git clone --branch mlperf_4.1 https://github.com/krai/axs $HOME/axs
echo "export PATH='$PATH:$HOME/axs'" >> $HOME/.bashrc
source $HOME/.bashrc
```

### [Optional] Refresh state
```
axs byname work_collection , remove
```

### Get required repositories and packages
```
axs byquery git_repo,collection,repo_name=axs2mlperf
axs byquery git_repo,collection,repo_name=axs2system
axs byquery git_repo,collection,repo_name=axs2kilt
axs byquery git_repo,collection,repo_name=axs2uai
axs byquery git_repo,collection,repo_name=kilt-mlperf
axs byquery git_repo,collection,repo_name=kilt4uai
```

## Setup the ImageNet dataset
### Download the dataset
Download the imagenet2012 (validation) dataset following the instructions from http://image-net.org/challenges/LSVRC/2012/, in accordance with https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection#datasets.

### Store the path to the dataset into an env variable
```
export IMAGENET_PATH=<paste here the path to the ImageNet dataset>
```
You should have the following folder structure:
<pre>
${IMAGENET_PATH}/imageNet/val/...
</pre>

## Run a Docker container

### Ensure that the current user is in the docker group
```
sudo usermod -aG docker ${USER}
```
### Locate the installer
The installation file **untether-speedai-buildable-containers.run** is distributed to all official purchasers of UntetherAI speedAI240 accelerators.

### Store the path to the installer and the target path into env variables:
```
export SOURCE_PATH=<paste here the path to your untether-speedai-buildable-containers.run>
export TARGET_PATH=<paste here the path where you want to install>
```

### Copy the installer
```
cp ${SOURCE_PATH}/untether-speedai-buildable-containers.run ${TARGET_PATH}
```
<pre>
$ md5sum untether-speedai-buildable-containers.run
66ab1e47e2bd03475da41a217ba0ac46
</pre>

### Run the installer
```
cd ${TARGET_PATH}
./untether-speedai-buildable-containers.run --noexec
```

### Build and start docker container
```
cd ${TARGET_PATH}/speedai-buildable-container
make
cd dockers/speedai_sdk
export DOCKER_EXTRA_ARGS="--volume ${HOME}/axs:$(echo ~)/axs \
--volume ${HOME}/work_collection:$(echo ~)/work_collection \
--volume ${IMAGENET_PATH}/imageNet:/server/datasets/imageNet"
./spinup.sh
```

### Install necessary packages
```
sudo apt update
sudo apt install -y autoconf
sudo apt install -y ntpdate
```

### Sync the time
```
sudo ln -snf /usr/share/zoneinfo/America/Toronto /etc/localtime
```

### Configure AXS
```
export PATH=$PATH:~/axs
```

## Run experiments

### Select one of the supported system-under-test (SUT)

| SUT | System name | System type | Board type | Board count |
|-----|-------------|-------------|------------|-------------|
| h13_u1_slim | Supermicro H13 | edge | slim | 1 |
| h13_u3_slim | Supermicro H13 | edge | slim | 3 |
| r760_u4_slim | Dell R760xa | edge | slim | 4 |
| r760_u6_slim | Dell R760xa | datacenter | slim | 6 |
| h13_u1_preview | Supermicro H13 | edge | preview | 1 |
| h13_u1_preview_dc | Supermicro H13 | datacenter | preview | 1 |
| h13_u2_preview | Supermicro H13 | edge | preview | 2 |
| h13_u2_preview_dc | Supermicro H13 | datacenter | preview | 2 |

```
export SUT=<selected SUT>
```

### Configure expected target QPS (Offline, Server) or latency (SingleStream, MultiStream)
```
export TARGET_QPS=<expected target QPS>
```
or
```
export TARGET_LATENCY=<expected target latency>
```

### Offline

#### Accuracy
```
axs byquery loadgen_output,task=image_classification,device=uai,framework=kilt,\
loadgen_scenario=Offline,sut_name=${SUT},loadgen_mode=AccuracyOnly,\
collection_name=experiments , get accuracy_report
```

#### Performance
```
axs byquery loadgen_output,task=image_classification,device=uai,framework=kilt,\
loadgen_scenario=Offline,sut_name=${SUT},loadgen_mode=PerformanceOnly,\
collection_name=experiments,loadgen_target_qps=${TARGET_QPS} , get performance
```

#### Performance with Power
```
axs byquery power_loadgen_output,task=image_classification,device=uai,framework=kilt,\
loadgen_scenario=Offline,sut_name=${SUT},loadgen_mode=PerformanceOnly,\
collection_name=experiments,loadgen_target_qps=${TARGET_QPS} , avg_power
```

### MultiStream

#### Accuracy
```
axs byquery loadgen_output,task=image_classification,device=uai,framework=kilt,\
loadgen_scenario=MultiStream,sut_name=${SUT},loadgen_mode=AccuracyOnly,\
collection_name=experiments , get accuracy_report
```

#### Performance
```
axs byquery loadgen_output,task=image_classification,device=uai,framework=kilt,\
loadgen_scenario=MultiStream,sut_name=${SUT},loadgen_mode=PerformanceOnly,\
collection_name=experiments,loadgen_target_latency=${TARGET_LATENCY} , get performance
```

#### Performance with Power
```
axs byquery power_loadgen_output,task=image_classification,device=uai,framework=kilt,\
loadgen_scenario=MultiStream,sut_name=${SUT},loadgen_mode=PerformanceOnly,\
collection_name=experiments,loadgen_target_latency=${TARGET_LATENCY} , avg_power
```

### SingleStream

#### Accuracy
```
axs byquery loadgen_output,task=image_classification,device=uai,framework=kilt,\
loadgen_scenario=SingleStream,sut_name=${SUT},loadgen_mode=AccuracyOnly,\
collection_name=experiments , get accuracy_report
```

#### Performance
```
axs byquery loadgen_output,task=image_classification,device=uai,framework=kilt,\
loadgen_scenario=SingleStream,sut_name=${SUT},loadgen_mode=PerformanceOnly,\
collection_name=experiments,loadgen_target_latency=${TARGET_LATENCY} , get performance
```

#### Performance with Power
```
axs byquery power_loadgen_output,task=image_classification,device=uai,framework=kilt,\
loadgen_scenario=SingleStream,sut_name=${SUT},loadgen_mode=PerformanceOnly,\
collection_name=experiments,loadgen_target_latency=${TARGET_LATENCY} , avg_power
```

### Server

#### Accuracy
```
axs byquery loadgen_output,task=image_classification,device=uai,framework=kilt,\
loadgen_scenario=Server,sut_name=${SUT},loadgen_mode=AccuracyOnly,\
collection_name=experiments , get accuracy_report
```

#### Performance
```
axs byquery loadgen_output,task=image_classification,device=uai,framework=kilt,\
loadgen_scenario=Server,sut_name=${SUT},loadgen_mode=PerformanceOnly,\
collection_name=experiments,loadgen_target_qps=${TARGET_QPS} , get performance
```

#### Performance with Power
```
axs byquery power_loadgen_output,task=image_classification,device=uai,framework=kilt,\
loadgen_scenario=Server,sut_name=${SUT},loadgen_mode=PerformanceOnly,\
collection_name=experiments,loadgen_target_qps=${TARGET_QPS} , avg_power
```