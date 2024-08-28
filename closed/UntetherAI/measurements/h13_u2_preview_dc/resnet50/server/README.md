
# MLPerf Inference v4.0 - closed - UntetherAI

To run experiments individually, use the following commands.

## h13_u2_preview - resnet50 - server

### Accuracy  

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=h13_u2_preview,device=uai,loadgen_mode=AccuracyOnly,loadgen_scenario=Server
```

### Performance 

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=h13_u2_preview,device=uai,loadgen_mode=PerformanceOnly,loadgen_compliance_test-,loadgen_scenario=Server,loadgen_target_qps=140300
```

### Compliance TEST01

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=h13_u2_preview,device=uai,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST01,loadgen_scenario=Server,loadgen_target_qps=140300
```

### Compliance TEST04

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=h13_u2_preview,device=uai,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST04,loadgen_scenario=Server,loadgen_target_qps=140300
```

### Compliance TEST05

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=h13_u2_preview,device=uai,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST05,loadgen_scenario=Server,loadgen_target_qps=140300
```

