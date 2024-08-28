
# MLPerf Inference v4.0 - closed - UntetherAI

To run experiments individually, use the following commands.

## r760_u6_slim - resnet50 - server

### Accuracy  

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=r760_u6_slim,device=uai,collection_name=experiments,loadgen_mode=AccuracyOnly,loadgen_scenario=Server
```

### Power 

```
axs byquery power_loadgen_output,framework=kilt,task=image_classification,sut_name=r760_u6_slim,device=uai,collection_name=experiments,loadgen_mode=PerformanceOnly,loadgen_compliance_test-,loadgen_scenario=Server,model_name=resnet50,loadgen_target_qps=310000
```

### Compliance TEST01

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=r760_u6_slim,device=uai,collection_name=experiments,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST01,loadgen_scenario=Server,model_name=resnet50,loadgen_target_qps=315000
```

### Compliance TEST04

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=r760_u6_slim,device=uai,collection_name=experiments,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST04,loadgen_scenario=Server,loadgen_target_qps=290000
```

### Compliance TEST05

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=r760_u6_slim,device=uai,collection_name=experiments,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST05,loadgen_scenario=Server,loadgen_target_qps=300000
```

