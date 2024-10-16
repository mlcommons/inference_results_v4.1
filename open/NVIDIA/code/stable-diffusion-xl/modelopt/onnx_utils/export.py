# Adapted https://github.com/huggingface/optimum/blob/15a162824d0c5d8aa7a3d14ab6e9bb07e5732fb6/optimum/exporters/onnx/convert.py#L573-L614

# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path

import onnx
import torch
from optimum.onnx.utils import _get_onnx_external_data_tensors, check_model_uses_external_data
from torch.onnx import export as onnx_export

# Per-tensor for INT8, we will convert it to FP8 later in onnxgraphsurgeon
SDXL_FP8_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8, "axis": None},
        "*input_quantizer": {"num_bits": 8, "axis": None},
        "*lm_head*": {"enable": False},
        "*output_layer*": {"enable": False},
        "default": {"num_bits": 8, "axis": None},
    },
    "algorithm": "max",
}


def generate_fp8_scales(vae):
    # temporary solution due to a known bug in torch.onnx._dynamo_export
    for _, module in vae.decoder.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            module.input_quantizer._num_bits = 8
            module.weight_quantizer._num_bits = 8
            module.input_quantizer._amax = (module.input_quantizer._amax * 127) / 448.0
            module.weight_quantizer._amax = (module.weight_quantizer._amax * 127) / 448.0


def generate_dummy_inputs(device):
    return torch.ones(1, 4, 128, 128).to(device)


def modelopt_export_sd(base, exp_name, model_name):
    os.makedirs(f"./{exp_name}", exist_ok=True)
    dummy_inputs = generate_dummy_inputs(device=base.vae.device)

    output = Path(f"{exp_name}/vae.onnx")
    input_names = ["latent"]
    output_names = ["images"]

    dynamic_axes = {"latent": {0: "B", 2: "H", 3: "W"}, "images": {0: "B", 2: "8H", 3: "8W"}}

    do_constant_folding = True
    opset_version = 17

    # Copied from Huggingface's Optimum
    # pipe.vae.forward = pipe.vae.decode
    onnx_export(
        base.vae,
        (dummy_inputs,),
        f=output.as_posix(),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=do_constant_folding,
        opset_version=opset_version,
    )

    onnx_model = onnx.load(str(output), load_external_data=False)
    model_uses_external_data = check_model_uses_external_data(onnx_model)

    if model_uses_external_data:
        tensors_paths = _get_onnx_external_data_tensors(onnx_model)
        onnx_model = onnx.load(str(output), load_external_data=True)
        onnx.save(
            onnx_model,
            str(output),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=output.name + "_data",
            size_threshold=1024,
        )
        for tensor in tensors_paths:
            os.remove(output.parent / tensor)
