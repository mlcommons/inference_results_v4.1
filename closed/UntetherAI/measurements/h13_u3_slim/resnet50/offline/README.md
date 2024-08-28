
# MLPerf Inference v4.0 - closed - UntetherAI

To run experiments individually, use the following commands.

## h13_u3_slim - resnet50 - offline

### Accuracy  

```
axs byquery loadgen_output,task=image_classification,device=uai,framework=kilt,loadgen_scenario=Offline,sut_name=h13_u3_slim,loadgen_mode=AccuracyOnly,collection_name=experiments,loadgen_min_duration_s=10,loadgen_buffer_size=50000
```

### Power 

```
axs byquery power_loadgen_output,framework=kilt,task=image_classification,sut_name=h13_u3_slim,device=uai,loadgen_mode=PerformanceOnly,loadgen_compliance_test-,loadgen_scenario=Offline,model_name=resnet50,governor_powersave_set+,DIMM=4,final+
```

### Compliance TEST01

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=h13_u3_slim,device=uai,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST01,loadgen_scenario=Offline,loadgen_target_qps=168000
```

### Compliance TEST04

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=h13_u3_slim,device=uai,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST04,loadgen_scenario=Offline,loadgen_target_qps=168000
```

### Compliance TEST05

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=h13_u3_slim,device=uai,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST05,loadgen_scenario=Offline,loadgen_target_qps=168000
```

