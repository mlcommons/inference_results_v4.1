
# MLPerf Inference v4.0 - closed - UntetherAI

To run experiments individually, use the following commands.

## h13_u1_preview - resnet50 - offline

### Accuracy  

```
axs byquery loadgen_output,task=image_classification,device=uai,framework=kilt,loadgen_scenario=Offline,sut_name=h13_u1_preview,loadgen_mode=AccuracyOnly,collection_name=experiments,loadgen_min_duration_s=10
```

### Performance 

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=h13_u1_preview,device=uai,loadgen_mode=PerformanceOnly,loadgen_compliance_test-,loadgen_scenario=Offline,loadgen_target_qps=70000
```

### Compliance TEST01

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=h13_u1_preview,device=uai,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST01,loadgen_scenario=Offline,loadgen_target_qps=70000
```

### Compliance TEST04

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=h13_u1_preview,device=uai,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST04,loadgen_scenario=Offline,loadgen_target_qps=70000
```

### Compliance TEST05

```
axs byquery loadgen_output,task=image_classification,device=uai,framework=kilt,loadgen_scenario=Offline,sut_name=h13_u1_preview,loadgen_mode=PerformanceOnly,collection_name=experiments,loadgen_min_duration_s=600,loadgen_compliance_test=TEST05,loadgen_target_qps=70000
```

