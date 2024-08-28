## MLPerf Inference Calibration and Quantization Details

## MLPerf Quantization in Closed Division Submissions:

### FP8 Quantization:
When quantizing Llama-2-70b, the following vLLM procedure was used. 
- [VLLM Offline Quantization with Static Activation Scaling Factors](https://docs.vllm.ai/en/latest/quantization/fp8.html#offline-quantization-with-static-activation-scaling-factors)
- Weights and activations were quantized
- KV cache was quantized

## Quantization in Open Division Submissions

### Int4 Quantization:
The bloke llama-2-70b-chat-GPTQ int4 quantized model, [Llama-2-70B-Chat-GPTQ](https://huggingface.co/TheBloke/Llama-2-70B-Chat-GPTQ), was used.   See details here: TheBloke/Llama-2-70B-Chat-GPTQ


Note: If applicable, for Open Division submissions, quantization details are in the READMEs attached to each individual Open Division submission.
