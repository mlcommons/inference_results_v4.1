# MLPerf Inference v4.1 Neural Magic Calibration Details

For all methods, the following LLM components are quantized:
- Linear layers (including dense and QKV linear)
- MLP Layer

## Quantization Methods

### 1. FP8 W8A8 (8-bit Float E4M3)

This method uses 8-bit floating-point quantization for both weights (W) and activations (A).
- Symmetric, per-tensor quantization
- Dynamic range: Max value observed in original precision on the calibration dataset
- Quantization formula: `x_q = round(clip(x / dr * m, -m, m))` where `dr` is the dynamic range, `m` is 448 (max of FP8 format)

Reference: [Llama-2-70b-chat-hf-FP8](https://huggingface.co/nm-testing/Llama-2-70b-chat-hf-FP8), [vLLM FP8 Quantization](https://docs.vllm.ai/en/latest/quantization/fp8.html)

### 2. INT8 W8A8 (8-bit Integer)

This method uses 8-bit integer quantization for both weights (W) and activations (A).
- Symmetric, per-channel quantization for weights
- Dynamic, per-token quantization for activations

### 3. GPTQ W4A16 (4-bit Weights, 16-bit Activations)

This method uses 4-bit quantization for weights (W) and no quantization for activations (A).
- Weights: 4-bit quantization with groupsize 128 using GPTQ
- Uses advanced quantization techniques to minimize accuracy loss

Reference: [Llama-2-70B-Chat-GPTQ](https://huggingface.co/nm-testing/Llama-2-70B-Chat-GPTQ)
