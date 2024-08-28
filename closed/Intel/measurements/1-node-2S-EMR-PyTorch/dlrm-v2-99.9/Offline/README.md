# DLRMv2 Inference on CPU

## LEGAL DISCLAIMER
To the extent that any data, datasets, or models are referenced by Intel or accessed using tools or code on this site such data, datasets and models are provided by the third party indicated as the source of such content. Intel does not create the data, datasets, or models, provide a license to any third-party data, datasets, or models referenced, and does not warrant their accuracy or quality. By accessing such data, dataset(s) or model(s) you agree to the terms associated with that content and that your use complies with the applicable license. 

Intel expressly disclaims the accuracy, adequacy, or completeness of any data, datasets or models, and is not liable for any errors, omissions, or defects in such content, or for any reliance thereon. Intel also expressly disclaims any warranty of non-infringement with respect to such data, dataset(s), or model(s). Intel is not liable for any liability or damages relating to your use of such data, datasets, or models. 

## Launch the Docker Image
Set the directories on the host system where model, dataset, and log files will reside. These locations will retain model and data content between Docker sessions.
```
export DATA_DIR="${DATA_DIR:-${PWD}/data}"
export MODEL_DIR="${MODEL_DIR:-${PWD}/model}"
export LOG_DIR="${LOG_DIR:-${PWD}/logs}"
```

## Launch the Docker Image
In the Host OS environment, run the following after setting the proper Docker image name. If the Docker image is not on the system already, it will be retrieved from the registry.

If retrieving the model or dataset, ensure any necessary proxy settings are run inside the container.
```
export DOCKER_IMAGE={INSERT_CONTAINER_NAME:TAG}

docker run --privileged -it --rm \
        --ipc=host --net=host --cap-add=ALL \
        -e http_proxy=${http_proxy} \
        -e https_proxy=${https_proxy} \
        -v ${DATA_DIR}:/data \
        -v ${MODEL_DIR}:/model \
        -v ${LOG_DIR}:/logs \
        --workdir /workspace \
        ${DOCKER_IMAGE} /bin/bash
```

## Download the Model [one-time operation]
Run this step inside the Docker container.  This is a one-time operation which will preserve the model on the host system using the volume mapping above.
```
bash download_model.sh
```

## Download the Dataset [one-time operation]
Run this step inside the Docker container.  This is a one-time operation which will preserve the dataset on the host system using the volume mapping above.
```
bash download_dataset.sh
```

## Calibrate the Model [one-time operation]
Run this step inside the Docker container.  This is a one-time operation, and the resulting calibrated model will be stored along with the original model file.
```
bash run_calibration.sh
```

## Run Benchmark
Run this step inside the Docker container.  Select the appropriate scenario.  If this is the first time running this workload, the original model file will be calibrated to INT8 and stored alongside the original model file (one-time operation).  The default configuration supports Intel EMR.  If running GNR, please make the following [additional changes](GNR.md).
```
SCENARIO=Offline ACCURACY=false bash run_mlperf.sh
SCENARIO=Server  ACCURACY=false bash run_mlperf.sh
SCENARIO=Offline ACCURACY=true  bash run_mlperf.sh
SCENARIO=Server  ACCURACY=true  bash run_mlperf.sh
```

## Run Compliance Tests
Run this step inside the Docker container.  After the benchmark scenarios have been run and results exist in {LOG_DIR}/results, run this step to complete compliance runs. Compliance output will be found in '{LOG_DIR}/compliance'.
```
bash run_compliance.sh
```

## Create Submission Content
Run this step inside the Docker container.  The following script will compile and structure the MLPerf Inference submission content into {LOG_DIR}, including 'code', 'calibration', 'measurements', and 'systems'. Ensure the system and measurement description files contained in '/workspace/descriptions' are correct before preceding.  Optionally pass 'CLEAR_CONTENT=true' to delete any existing 'code', 'calibration', and 'measurements' content before populating.
```
# Ensure the correctness of '/workspace/descriptions/systems'.
# Ensure the correctness of '/workspace/descriptions/measurements'.
bash populate_submission.sh

# [Optional] Alternatively:
# CLEAR_CONTENT=true bash populate_submission.sh
```

## Validate Submission Checker
Run this step inside the Docker container.  The following script will perform accuracy log truncation and run the submission checker on the contents of {LOG_DIR}. The source scripts are distributed as MLPerf Inference reference tools. Ensure the submission content has been populated before running.  The script output is transient and destroyed after running.  The original content of ${LOG_DIR} is not modified.
```
bash run_submission_checker.sh
```
