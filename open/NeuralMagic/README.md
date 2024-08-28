# Neural Magic MLPerf Inference v4.1 Submission

This is the repository of [Neural Magic's](https://neuralmagic.com/) submission for [MLPerf Inference Benchmark v4.1](https://mlcommons.org/benchmarks/inference-datacenter/).

This round's submission features LLM benchmarks faciliated by [Collective Mind (MLCommons CM)](https://github.com/mlcommons/ck/) across GPU architectures, model architectures, and optimization methods with models from [Neural Magic's model zoo](https://huggingface.co/neuralmagic) running on vLLM. CM provides a universal interface to any software project and transforms it into a database of reusable automation actions and portable scripts in a transparent and non-intrusive way.

## Calibration Details

For all methods, the following LLM components are quantized:
- Linear layers (including dense and QKV linear)
- MLP Layer

### FP8 W8A8 (8-bit Float E4M3)

Command to start vLLM server on 4xH100:
```bash
vllm serve nm-testing/Llama-2-70b-chat-hf-FP8 --tensor-parallel-size=4 --max-num-seqs=1024 --max-model-len=2048 --enable-chunked-prefill --max-num-batched-tokens=2048 --gpu-memory-utilization=0.95 --disable-log-requests
```

This method uses 8-bit floating-point quantization for both weights (W) and activations (A).
- Symmetric, per-tensor quantization
- Dynamic range: Max value observed in original precision on the calibration dataset
- Quantization formula: `x_q = round(clip(x / dr * m, -m, m))` where `dr` is the dynamic range, `m` is 448 (max of FP8 format)

Reference: [Llama-2-70b-chat-hf-FP8](https://huggingface.co/nm-testing/Llama-2-70b-chat-hf-FP8), [vLLM FP8 Quantization](https://docs.vllm.ai/en/latest/quantization/fp8.html)

### GPTQ W4A16 (4-bit Weights, 16-bit Activations)

Command to start vLLM server on 4xH100:
```bash
vllm serve nm-testing/Llama-2-70B-Chat-GPTQ --tensor-parallel-size=4 --max-num-seqs=1024 --max-model-len=2048 --enable-chunked-prefill --max-num-batched-tokens=2048 --gpu-memory-utilization=0.95 --disable-log-requests
```

This method uses 4-bit quantization for weights (W) and no quantization for activations (A).
- Weights: 4-bit quantization with groupsize 128 using GPTQ
- Uses advanced quantization techniques to minimize accuracy loss

Reference: [Llama-2-70B-Chat-GPTQ](https://huggingface.co/nm-testing/Llama-2-70B-Chat-GPTQ)