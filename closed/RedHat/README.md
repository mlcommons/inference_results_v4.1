# MLPerf Inference v4.1 Red Hat Inc


This is a repository for Red Hat's submission to the MLPerf Inference v4.1 benchmark.  It includes implementations of the benchmark code written to utilize the vLLM runtime via it's implementation of the OpenAI API. The vLLM runtime was run on Red Hat OpenShift Container Platform with the OpenShift AI model serving stack based on the kserve project.

# Contents

Each model implementation in the `code` subdirectory has:

* Code that implements inferencing
* A Dockerfile which can be used to build a container for the benchmark
* Documentation on the dataset, model, and machine setup

# Hardware & Software requirements

These benchmarks have been tested on the following machine configuration:

* A Dell Inc. PowerEdge R760xa with 4x NVIDIA L40S GPUs
* A Dell Inc. XE9680 server with 8x NVIDIA H100-SXM-80GB GPUs


The required software stack includes:
* [Red Hat OpenShift](https://access.redhat.com/documentation/en-us/openshift_container_platform/4.15/html/installing/index)
*  [Node Feature Discovery Operator](https://docs.nvidia.com/launchpad/infrastructure/openshift-it/latest/openshift-it-step-01.html)
* [Nvidia GPU Operator](https://docs.nvidia.com/launchpad/infrastructure/openshift-it/latest/openshift-it-step-03.html)
* [OpenShift AI model serving stack](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/2.11/html/serving_models/serving-large-models_serving-large-models#configuring-automated-installation-of-kserve_serving-large-models)
  - Red Hat Service Mesh Operator
  - Red Hat Serverless Operator
  - OpenShift AI with KServe component enabled.

Each benchmark can be run with the following steps:

1. Follow the instructions in the README in the api-endpoint-artifacts directory to deploy the model, deploy the test Pod, and run the test in the test Pod with `oc exec`.
Collapse





