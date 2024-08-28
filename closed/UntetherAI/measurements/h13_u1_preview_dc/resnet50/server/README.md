
# MLPerf Inference v4.0 - closed - UntetherAI

To run experiments individually, use the following commands.

## h13_u1_preview_dc - resnet50 - server

### Accuracy  

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=h13_u1_preview_dc,device=uai,loadgen_mode=AccuracyOnly,loadgen_scenario=Server
```

### Performance 

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=h13_u1_preview_dc,device=uai,loadgen_mode=PerformanceOnly,loadgen_scenario=Server,loadgen_target_qps=70100,speedai_devices=uaia*
```

### Compliance TEST01

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=h13_u1_preview_dc,device=uai,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST01,loadgen_scenario=Server,loadgen_target_qps=70000,speedai_devices=uaia*
```

### Compliance TEST04

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=h13_u1_preview_dc,device=uai,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST04,loadgen_scenario=Server,loadgen_target_qps=70000,speedai_devices=uaia*
```

### Compliance TEST05

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=h13_u1_preview_dc,device=uai,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST05,loadgen_scenario=Server,loadgen_target_qps=70000,speedai_devices=uaia*
```

