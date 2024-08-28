# MLPerf Inference v4.1 - Untether AI - Calibration Details

## ResNet50

### Calibration

We pass a set of images through the network (specifically, those specified in `cal_image_list_option_1.txt`), recording the minimum and maximum observed values for all tensors.

### Quantization

All activations and weights are quantized identically:

- The data type is fixed to be `FP8p` (see [whitepaper](https://www.untether.ai/download-resource/?label=FP8+Whitepaper)).
- The quantization scheme follows the standard formula `dequant(q) = (q - Z) / S`, with the addition restrictions of symmetry (`Z = 0`) and power-of-two scaling (`S = 2^b`, where `b` is an integer).
- The quantization parameter `b` is chosen to be maximal such that the ranges observed during calibration are representable.
- Each tensor gets a single quantization parameter (i.e. no per-channel scaling).
