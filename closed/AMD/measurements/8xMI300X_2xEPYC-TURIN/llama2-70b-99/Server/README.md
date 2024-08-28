# Run llama2-70b-99.9 Server
## Setup
Start an inference Docker container using the instructions in /closed/AMD/code/llama2-70b-99.9/README.md.

## Run Benchmarks
In the container, use the scripts in /closed/AMD/code/llama2-70b-99.9/test_VllmFp8 to run the benchmarks.

### Run Accuracy
``` bash
cd /lab-hist/submission-package-20240725c/closed/AMD/code/llama2-70b-99.9/test_VllmFp8
SUBMISSION=1 ./test_VllmFp8_SyncServer_acc.sh
```

### Run Performance
``` bash
cd /lab-hist/submission-package-20240725c/closed/AMD/code/llama2-70b-99.9/test_VllmFp8
SUBMISSION=1 ./test_VllmFp8_SyncServer_perf.sh
```

## Results
In the container, results are written to `/lab-hist/mlperf-results/$datetime1/$datetime2/Server`.