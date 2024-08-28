
# MLPerf Inference v4.1 - open - Krai

To run experiments individually, use the following commands.

## 7920t-kilt-onnxruntime_cpu - resnet50 - multistream

### Accuracy  

```
axs byquery sut_name=7920t-kilt-onnxruntime_cpu,loadgen_output,task=image_classification,device=onnxrt,backend_type=gpu,loadgen_scenario=MultiStream,framework=kilt,model_name=resnet50,loadgen_mode=AccuracyOnly,collection_name=experiments_cpu,loadgen_dataset_size=50000,loadgen_buffer_size=1024
```

### Performance 

```
axs byquery sut_name=7920t-kilt-onnxruntime_cpu,loadgen_output,task=image_classification,device=onnxrt,backend_type=cpu,loadgen_scenario=MultiStream,framework=kilt,model_name=resnet50,loadgen_mode=PerformanceOnly,collection_name=experiments_cpu,loadgen_dataset_size=50000,loadgen_buffer_size=1024,loadgen_target_latency=0.42
```

