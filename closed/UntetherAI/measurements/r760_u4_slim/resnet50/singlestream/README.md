
# MLPerf Inference v4.0 - closed - UntetherAI

To run experiments individually, use the following commands.

## r760_u4_slim - resnet50 - singlestream

### Accuracy  

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=r760_u4_slim,device=uai,loadgen_mode=AccuracyOnly,loadgen_scenario=SingleStream
```

### Performance 

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=r760_u4_slim,device=uai,loadgen_mode=PerformanceOnly,loadgen_compliance_test-,loadgen_scenario=SingleStream,loadgen_target_latency=0.12
```

### Compliance TEST01

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=r760_u4_slim,device=uai,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST01,loadgen_scenario=SingleStream,loadgen_target_latency=0.12
```

### Compliance TEST04

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=r760_u4_slim,device=uai,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST04,loadgen_scenario=SingleStream,loadgen_target_latency=0.12
```

### Compliance TEST05

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=r760_u4_slim,device=uai,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST05,loadgen_scenario=SingleStream,loadgen_target_latency=0.12
```

