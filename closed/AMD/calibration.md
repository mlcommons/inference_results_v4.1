# Llama2-70b

LLaMA-2-70B is quantized to OCP FP8-e4m3 in Quark framework using the following approach.

## Calibration Dataset

We utilize 1000 samples from open-orca and pre-process them according to mlcommons/inference standards to create our calibration dataset. 

##LLaMA Model

The LLaMA-2-70B-chat-hf model, initially in FP16 format, is downloaded from Hugging Face for quantization. 

## FP8 Quantization

### FP8 Quantization Process:

We apply per-tensor symmetric static quantization for OCP FP8-e4m3 to quantize LLaMA-2-70B. The quantized value xq is computed from the original value x as:

x_q = round( clip (x / scale * 448, -448, 448))

where “scale” is the maximum absolute value (amax) of x, 448 represents the range of OCP FP8-e4m3, and scaled value is rounded using the half-even method after clipping.

### Quantization Strategy:

All nn.linear modules within the decoder blocks of the model are quantized, including inputs and weights. Quantization scales of inputs and weights are computed “statically”, that is they are computed fully during the calibration step, before used in inference. The weights of Q, K, V share an identical quantization scale which is the largest value among the three separate weight scales. KV cache entries are quantized, and K and V share an identical quantization scale, which is computed during calibration.

Summarizing the quantized tensors and key configurations:
 - Inputs and weights of linear modules within the decoder blocks are quantized.
 - Quantization scales of inputs and weights are computed during calibration.
 - Q, K, V share an identical weight quantization scale.
 - KV cache entries are quantized, and K and V share an identical quantization scale computed during calibration.

## Quantize LLaMA using Quark

(1) Download Quark whl from [here](https://www.xilinx.com/bin/public/openDownload?filename=quark-0.1.0+a9827f5-py39-none-any.whl) and install Quark
(2) Quantize LLaMA model:
cd examples/torch/language_modeling/
python3 quantize_quark.py --model_dir $model_dir \
                          --output_dir $output_dir \
                          --quant_scheme w_fp8_a_fp8_o_fp8 \
                          --dataset $calib_dataset \
                          --num_calib_data 1000 \
                          --model_export vllm_adopted_safetensors \
                          --no_weight_matrix_merge



# SDXL Calibration and Quantization

This document contains information about how SDXL was quantized for an MLPerf inference submission for v4.1.
SDXL was quantized heterogeneously with Int8/FP8/FP16 quantization, which is detailed herein.
Quantization of SDXL for this submission is achieved by leveraging the library [Brevitas](https://github.com/xilinx/brevitas).
If something is not clear, please refer to the quantization reproduction scripts.
If something is still unclear, please post an issue on [Brevitas's GitHub](https://github.com/xilinx/brevitas).

## Quantization Methodology

At a high level, the entire quantization pipeline can be described as follows:
 1. SmoothQuant is applied to all the Linear & Conv2d layers of UNet which are subject to quantization
 2. Quantization nodes are inserted into UNet around specific operators, as described in [below](#insertion-of-quantization-nodes)
 3. Calibration is applied, as described [below](#calibration-method)
 4. VAE is adapted to work with FP16, as described [below](#vae)
 5. Weights and quantization parameters of UNet and VAE are exported, to be used by a downstream inference toolchain.

The specifics needed to know for each step are described below.

## Quantization Functions

SDXL is quantized with Int8, FP8 & FP16 - details on which are described below.
For Int8/FP8, so-called "fake quantization" is used during the quantization,
where "quantize/dequantize" functions are applied to the operands of a given function.
Note, the quantization functions are parametrized by their associated `scale` and `zero_point`,
the calculation of which is described in the [Calibration Method section](#calibration-method).
Note, all quantization functions assume PyTorch has been imported through `import torch`.

### Int8

The int8 quantization function can be characterized by the following `quantize_int8`/`dequantize_int8` functions:

```python
def quantize_int8(tensor: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, is_asym: bool):
    if is_asym:
        clamp_min, clamp_max = torch.tensor(0.), torch.tensor(255.)
    else:
        clamp_min, clamp_max = torch.tensor(-128.), torch.tensor(127.)
    quant_tensor = torch.clamp(torch.round(tensor/scale + zero_point), clamp_min, clamp_max) 
    return quant_tensor
```

```python
def dequantize_int8(tensor: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor):
    return (tensor - zero_point) * scale
```

Where `scale.dtype = torch.float16` and `zero_point.dtype = torch.int8`.

### FP8

In this work, the `fp8_e4m3fnuz` is used throughout, whose quantization function can be characterized by the following `quantize_fp8`/`dequantize_fp8` functions:

```python
def quantize_fp8(tensor: torch.Tensor, scale: torch.Tensor):
    clamp_min, clamp_max = torch.tensor(-240), torch.tensor(240)
    quant_tensor = torch.clamp((tensor/scale).to(torch.float8_e4m3fnuz).to(tensor.dtype), clamp_min, clamp_max)
    return quant_tensor
```

```python
def dequantize_fp8(tensor: torch.Tensor, scale: torch.Tensor):
    return tensor * scale
```

Where `scale.dtype = torch.float16`.

### FP16

When FP16 is employed, no `quantize`/`dequantize` functions are necessary,
we use FP16 with a straight-forward conversion from FP32, e.g., `tensor.to(torch.float16)`.

## Insertion of Quantization Nodes

Now that we've defined our quantization functions,
the insertion these functions within SDXL's compute graph fully describe how quantized SDXL is quantized.
We only insert quantization nodes in the UNet "sub-network" of SDXL around linear (i.e., fully connected), conv2d and attention layers.
When Int8 quantization is used, it typically has the following properties:
 - Dynamically-calculated tensors (e.g., activations and layer inputs):
   - tensor-wide scaling (i.e., `scale` is a scalar)
   - symmetric (i.e., `asym=False`, `zero_point=0`)
 - Parameters (e.g., weights):
   - output-channel scaling (i.e., `scale` is a vector with the same number of elements as the output channels of a given layer, and is "broadcastable" with the layer's weight tensor)
   - asymmetric (i.e., `asym=False`, `zero_point` is a vector with the same number of elements as the output channels of a given layer, and is "broadcastable" with the layer's weight tensor

When FP8 quantization is used, it typically has the following properties:
 - Dynamically-calculated tensors (e.g., activations and layer inputs):
   - tensor-wide scaling (i.e., `scale` is a scalar)
 - Parameters (e.g., weights):
   - output-channel scaling (i.e., `scale` is a vector with the same number of elements as the output channels of a given layer, and is "broadcastable" with the layer's weight tensor)

### Linear Layers

When linear layers in the UNet portion of SDXL are quantized at their inputs & weights.
Refer to the quantization reproduction script to see which linear layers within UNet are quantized.
Typically, the linear layers are quantized to Int8 precision.

### Convolutional Layers

All convolutional layers in the UNet portion of SDXL are quantized at their inputs & weights.
Refer to the quantization reproduction script to see which convolutional layers within UNet are quantized.
Typically, the convolutional layers are quantized to Int8 precision.

### Attention Layers

The attention layers have some careful consideration, in general, the linear layers within attention are quantized (as [above](#linear-layers)), while the input to the other matrix multiplies (i.e., the ones in the `torch.Functional.scaled_dot_product_attention`) may also be quantized.
Refer to the quantization reproduction script to see which attention layers within UNet are quantized.
Typically, the linear layers within attention are quantized to Int8,
while the other matrix multiplies may be FP16 or FP8.

## Calibration Method

Calibration is performed on a subset of the 500 calibration prompts provided by MLPerf.
Refer to the quantization reproduction script to see which specific prompts were used.

### Weights

The scale vector is chosen for each weight tensor as follows,

```python
scale = torch.max(torch.abs(weight + weight_bias),dim=1,keep_dim=1).values / max_val
```

where `weight_bias` and `max_val` have different values, under the following scenarios:
 - when Int8 is used and `asym=True`, `weight_bias=-torch.min(weight,dim=1,keep_dim=1).values`, `max_val=255`;
 - when Int8 is used and `asym=False`, `weight_bias=0`, `max_val=128`; and
 - when FP8 is used, `weight_bias=0`, `max_val=240`.

### Activations

The scale scalar is chosen for each dynamically-calculated tensor,
such that the largest magnitude value of the tensor in any of the selected calibration prompts can be represented exactly without overflow.

Explicity, calculated as follows:

```python
scale = torch.max(torch.abs(input)) / max_val
```

Where `max_val = 128` for Int8 and `max_val = 240` for FP8.

### Post-Training Quantization Techniques

Several post-training quantization (PTQ) techniques have been applied to improve the accuracy of the quantized model,
including bias correction, GPTQ and [SmoothQuant](https://arxiv.org/abs/2211.10438).

#### Bias Correction

Bias correction may be applied to a number of layers.
This modifies the bias of those layers such that the mean of the output distributions of those layers more closely matches the unquantized model.
Please see the quantization reproduction script to find which layers had bias correction applied to them.

#### GPTQ

[GPTQ](https://arxiv.org/abs/2210.17323) may be applied to a number of layers.
This modifies the weights of those layers to minimise the MSE error between the quantized and unquantized model.
Please see the quantization reproduction script to find which layers had GPTQ applied to them.

### Guidance Scale

During quantization, we performed hyperparameter optimization of the guidance scale,
to attain one optimal for our quantization scheme.
Refer to the quantization reproduction script to see what value for guidance scale was used.

### VAE

The weights and biases of several of the convolutional layers of VAE may also be scaled by 1/128 in order to avoid overflow during FP16 compute.
Refer to the quantization reproduction script to see what specific layers were scaled in this way.
