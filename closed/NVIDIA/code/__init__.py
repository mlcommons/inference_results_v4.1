#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from collections import namedtuple
from importlib import import_module
from typing import Dict

from code.common import logging
from code.common.constants import AliasedNameEnum, Benchmark

ResNet50Component = import_module("code.resnet50.tensorrt.constants").ResNet50Component
RetinanetComponent = import_module("code.retinanet.tensorrt.constants").RetinanetComponent
BERTComponent = import_module("code.bert.tensorrt.constants").BERTComponent
DLRMv2Component = import_module("code.dlrm-v2.tensorrt.constants").DLRMv2Component
UNET3DComponent = import_module("code.3d-unet.tensorrt.constants").UNET3DComponent
GPTJComponent = import_module("code.gptj.tensorrt.constants").GPTJComponent
LLAMA2Component = import_module("code.llama2-70b.tensorrt.constants").LLAMA2Component
Mixtral8x7BComponent = import_module("code.mixtral-8x7b.tensorrt.constants").Mixtral8x7BComponent
SDXLComponent = import_module("code.stable-diffusion-xl.tensorrt.constants").SDXLComponent

G_BENCHMARK_COMPONENT_ENUM_MAP: Dict[Benchmark, AliasedNameEnum] = {
    Benchmark.BERT: BERTComponent,
    Benchmark.GPTJ: GPTJComponent,
    Benchmark.LLAMA2: LLAMA2Component,
    Benchmark.Mixtral8x7B: Mixtral8x7BComponent,
    Benchmark.DLRMv2: DLRMv2Component,
    Benchmark.ResNet50: ResNet50Component,
    Benchmark.Retinanet: RetinanetComponent,
    Benchmark.UNET3D: UNET3DComponent,
    Benchmark.SDXL: SDXLComponent,
}
G_BENCHMARK_ACCELERATOR_COMPONENT_ALIAS_MAP = {
    Benchmark.ResNet50: {"gpu": [{ResNet50Component.ResNet50},
                                 {ResNet50Component.PreRes2, ResNet50Component.Res2Res3, ResNet50Component.Res3},
                                 {ResNet50Component.PreRes3, ResNet50Component.Res3, ResNet50Component.PostRes3}],
                         "dla": [{ResNet50Component.ResNet50},
                                 {ResNet50Component.Backbone, ResNet50Component.TopK}]},
    Benchmark.Retinanet: {"gpu": [{RetinanetComponent.Retinanet}],
                          "dla": [{RetinanetComponent.Retinanet},
                                  {RetinanetComponent.Backbone, RetinanetComponent.NMS}]},
    Benchmark.BERT: {"gpu": [{BERTComponent.BERT}]},
    Benchmark.DLRMv2: {"gpu": [{DLRMv2Component.DLRMv2}]},
    Benchmark.UNET3D: {"gpu": [{UNET3DComponent.UNET3D}]},
    Benchmark.GPTJ: {"gpu": [{GPTJComponent.GPTJ}]},
    Benchmark.LLAMA2: {"gpu": [{LLAMA2Component.LLAMA2}]},
    Benchmark.Mixtral8x7B: {"gpu": [{Mixtral8x7BComponent.Mixtral8x7B}]},
    Benchmark.SDXL: {"gpu": [{SDXLComponent.CLIP1, SDXLComponent.CLIP2, SDXLComponent.UNETXL, SDXLComponent.VAE}]},
}

# Instead of storing the objects themselves in maps, we store object locations, as we do not want to import redundant
# modules on every run. Some modules include CDLLs and TensorRT plugins, or have large imports that impact runtime.
# Dynamic imports are also preferred, as some modules (due to their legacy model / directory names) include dashes.
ModuleLocation = namedtuple("ModuleLocation", ("module_path", "cls_name"))
G_BENCHMARK_CLASS_MAP = {
    Benchmark.ResNet50: ModuleLocation("code.resnet50.tensorrt.builder", "ResNet50"),
    Benchmark.Retinanet: ModuleLocation("code.retinanet.tensorrt.builder", "Retinanet"),
    Benchmark.BERT: ModuleLocation("code.bert.tensorrt.builder", "BERT"),
    Benchmark.DLRMv2: ModuleLocation("code.dlrm-v2.tensorrt.builder", "DLRMv2"),
    Benchmark.UNET3D: ModuleLocation("code.3d-unet.tensorrt.builder", "UnetBuilder"),
    Benchmark.GPTJ: ModuleLocation("code.gptj.tensorrt.builder", "GPTJ6B"),
    Benchmark.LLAMA2: ModuleLocation("code.llama2-70b.tensorrt.builder", "LLAMA2"),
    Benchmark.Mixtral8x7B: ModuleLocation("code.mixtral-8x7b.tensorrt.builder", "MIXTRAL8x7B"),
    Benchmark.SDXL: ModuleLocation("code.stable-diffusion-xl.tensorrt.builder", "SDXL"),
}
G_HARNESS_CLASS_MAP = {
    "triton_harness": ModuleLocation("code.common.server_harness", "TritonHarness"),
    "triton_llama_harness": ModuleLocation("code.llama2-70b.tensorrt.triton_harness", "TritonLlamaHarness"),
    "bert_harness": ModuleLocation("code.bert.tensorrt.harness", "BertHarness"),
    "gptj_harness": ModuleLocation("code.gptj.tensorrt.harness", "GPTJHarness"),
    "llama2_harness": ModuleLocation("code.llama2-70b.tensorrt.harness", "LLAMA2Harness"),
    "mixtral8x7b_harness": ModuleLocation("code.mixtral-8x7b.tensorrt.harness", "Mixtral8x7BHarness"),
    "dlrm_v2_harness": ModuleLocation("code.dlrm-v2.tensorrt.harness", "DLRMv2Harness"),
    "unet3d_harness": ModuleLocation("code.3d-unet.tensorrt.harness", "UNet3DKiTS19Harness"),
    "sdxl_harness": ModuleLocation("code.stable-diffusion-xl.tensorrt.harness", "SDXLHarness"),
    "lwis_harness": ModuleLocation("code.common.lwis_harness", "LWISHarness"),
    "profiler_harness": ModuleLocation("code.internal.profiler", "ProfilerHarness"),
}


def get_cls(module_loc: ModuleLocation) -> type:
    """
    Returns the specified class denoted by a ModuleLocation.

    Args:
        module_loc (ModuleLocation):
            The ModuleLocation to specify the import path of the class

    Returns:
        type: The imported class located at module_loc
    """
    return getattr(import_module(module_loc.module_path), module_loc.cls_name)


def validate_batch_size(conf):
    # triton does not support batch splitting, enforce a single engine component
    if conf.get("use_triton") and len(conf.get("gpu_batch_size")) != 1:
        raise ValueError(f"triton harness does not support multi component inference, batch_size must have only 1 entry instead of {len(conf.get('batch_size'))}: {conf.get('batch_size')}")

    benchmark = conf["benchmark"]
    # check if gpu_batch_size and dla_batch_size is of a valid component combination
    for accelerator in ("gpu", "dla"):
        batch_size_dict = conf.get(f"{accelerator}_batch_size")
        # If not using this accelerator then continue
        if not batch_size_dict:
            continue
        if not any(set(batch_size_dict.keys()) == valid_components for valid_components in G_BENCHMARK_ACCELERATOR_COMPONENT_ALIAS_MAP[benchmark][accelerator]):
            raise ValueError(f"[{benchmark.valstr()}] {accelerator}_batch_size : {batch_size_dict} does not have supported component combinations. Valid combinations are \
                             {[valid_components for valid_components in G_BENCHMARK_ACCELERATOR_COMPONENT_ALIAS_MAP[benchmark][accelerator]]}")


def convert_to_component_aliased_enum(conf):
    benchmark = conf["benchmark"]

    for accelerator in ("gpu", "dla"):
        batch_size_dict = conf.get(f"{accelerator}_batch_size")
        # If not using this accelerator then continue
        if not batch_size_dict:
            continue
        for component in list(batch_size_dict.keys()):
            component_alias_enum = G_BENCHMARK_COMPONENT_ENUM_MAP[benchmark].get_match(component)
            if not component_alias_enum:
                raise ValueError(f"[{benchmark.valstr()}] {accelerator}_batch_size : {batch_size_dict} has unsupported component {component}")
            batch_size_dict[component_alias_enum] = batch_size_dict.pop(component)


def get_benchmark(conf):
    """Return module of benchmark initialized with config."""

    benchmark = conf["benchmark"]
    if not isinstance(benchmark, Benchmark):
        logging.warning(f"'benchmark: {benchmark}' in config is not Benchmark Enum member. This behavior is deprecated.")
        benchmark = Benchmark.get_match(benchmark)
        if benchmark is None:
            ttype = type(conf["benchmark"])
            raise ValueError(f"'benchmark' in config is not of supported type '{ttype}'")

    if benchmark not in G_BENCHMARK_CLASS_MAP:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    convert_to_component_aliased_enum(conf)
    # batch size validity check
    validate_batch_size(conf)

    builder_op = get_cls(G_BENCHMARK_CLASS_MAP[benchmark])
    return builder_op(conf)


def get_harness(conf, profile):
    """Refactors harness generation for use by functions other than handle_run_harness."""

    benchmark = conf["benchmark"]
    if not isinstance(benchmark, Benchmark):
        logging.warning("'benchmark' in config is not Benchmark Enum member. This behavior is deprecated.")
        benchmark = Benchmark.get_match(benchmark)
        if benchmark is None:
            ttype = type(conf["benchmark"])
            raise ValueError(f"'benchmark' in config is not of supported type '{ttype}'")

    convert_to_component_aliased_enum(conf)
    # batch size validity check
    validate_batch_size(conf)

    if "triton_unified" in conf.get("config_ver"):
        k = "triton_unified_harness"
        conf["inference_server"] = "triton"
    elif conf.get("use_triton"):
        k = "triton_harness"
        if Benchmark.LLAMA2 == benchmark:
            k = "triton_llama_harness"
        conf["inference_server"] = "triton"
    elif Benchmark.BERT == benchmark:
        k = "bert_harness"
        conf["inference_server"] = "custom"
    elif Benchmark.GPTJ == benchmark:
        k = "gptj_harness"
        conf["inference_server"] = "custom"
    elif Benchmark.LLAMA2 == benchmark:
        k = "llama2_harness"
        conf["inference_server"] = "custom"
    elif Benchmark.Mixtral8x7B == benchmark:
        k = "mixtral8x7b_harness"
        conf["inference_server"] = "custom"
    elif Benchmark.DLRMv2 == benchmark:
        k = "dlrm_v2_harness"
        conf["inference_server"] = "custom"
    elif Benchmark.UNET3D == benchmark:
        k = "unet3d_harness"
        conf["inference_server"] = "custom"
    elif Benchmark.SDXL == benchmark:
        k = "sdxl_harness"
        conf["inference_server"] = "custom"
    else:
        k = "lwis_harness"
        conf["inference_server"] = "lwis"

    harness = get_cls(G_HARNESS_CLASS_MAP[k])(conf, benchmark)

    # Attempt to run profiler. Note that this is only available internally.
    if profile is not None:
        try:
            harness = get_cls(G_HARNESS_CLASS_MAP["profiler_harness"])(harness, profile)
        except BaseException:
            logging.info("Could not load profiler: Are you an internal user?")

    return harness, conf
