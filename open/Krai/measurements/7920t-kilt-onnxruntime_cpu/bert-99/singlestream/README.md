
# MLPerf Inference v4.1 - open - Krai

To run experiments individually, use the following commands.

## 7920t-kilt-onnxruntime_cpu - bert-99 - singlestream

### Accuracy  

```
axs byquery sut_name=7920t-kilt-onnxruntime_cpu,loadgen_output,task=bert,device=onnxrt,backend_type=cpu,framework=kilt,model_name=bert-99,loadgen_scenario=SingleStream,loadgen_mode=AccuracyOnly,loadgen_dataset_size=10833,loadgen_buffer_size=10833,collection_name=experiments_cpu_bert
```

### Performance 

```
axs byquery sut_name=7920t-kilt-onnxruntime_cpu,loadgen_output,task=bert,device=onnxrt,backend_type=cpu,framework=kilt,model_name=bert-99,loadgen_scenario=SingleStream,loadgen_mode=PerformanceOnly,loadgen_dataset_size=10833,loadgen_buffer_size=10833,collection_name=experiments_cpu_bert
```

