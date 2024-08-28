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

import os
import sys
sys.path.insert(0, os.getcwd())

from importlib import import_module
from code.common.constants import Benchmark, Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *

ParentConfig = import_module("configs.llama2-70b")
GPUBaseConfig = ParentConfig.GPUBaseConfig


class SingleStreamGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.SingleStream
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    tensor_parallelism = 1
    gpu_batch_size = {'llama2-70b': 1}
    precision = "fp16"
    enable_sort = False
    kvcache_free_gpu_mem_frac = 0.9


################## open division configs start from here ########################

class SingleStreamGPUOpenBaseConfig(SingleStreamGPUBaseConfig):
    model_path = "build/open-models/LLAMA2-70B/"
    gpu_batch_size = {'llama2-70b': 1}
    gpu_copy_streams = 1
    gpu_inference_streams = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, ModelOpt.DepthPruned)
class H100_SXM_80GBx1(SingleStreamGPUOpenBaseConfig):
    system = KnownSystem.H100_SXM_80GBx1
    single_stream_expected_latency_ns = 1876267940
    model_path = "build/open-models/LLAMA2-70B-DepthPruned/"
    quant_model_path = "FP8_quantized"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP, ModelOpt.DepthPruned)
class H100_SXM_80GBx1_HighAccuracy(H100_SXM_80GBx1):
    single_stream_expected_latency_ns = 1876267940


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, ModelOpt.DepthPruned)
class H200_SXM_141GBx1(SingleStreamGPUOpenBaseConfig):
    system = KnownSystem.H200_SXM_141GBx1
    single_stream_expected_latency_ns = 1876267940
    model_path = "build/open-models/LLAMA2-70B-DepthPruned/"
    quant_model_path = "FP8_quantized"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP, ModelOpt.DepthPruned)
class H200_SXM_80GBx1_HighAccuracy(H200_SXM_141GBx1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, ModelOpt.DepthPruned)
class Orin(SingleStreamGPUOpenBaseConfig):
    system = KnownSystem.Orin
    model_path = "build/open-models/LLAMA2-70B-DepthPruned/"
    quant_model_path = "INT4AWQ_quantized"
    single_stream_expected_latency_ns = 5000000000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP, ModelOpt.DepthPruned)
class Orin_HighAccuracy(Orin):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, ModelOpt.DepthPrunedMedusa)
class H100_SXM_80GBx1(SingleStreamGPUOpenBaseConfig):
    system = KnownSystem.H100_SXM_80GBx1
    single_stream_expected_latency_ns = 1876267940
    model_path = "build/open-models/LLAMA2-70B-DepthPrunedMedusa/"
    quant_model_path = "INT4AWQ_quantized"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP, ModelOpt.DepthPrunedMedusa)
class H100_SXM_80GBx1_HighAccuracy(H100_SXM_80GBx1):
    single_stream_expected_latency_ns = 1876267940


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, ModelOpt.DepthPrunedMedusa)
class H200_SXM_141GBx1(SingleStreamGPUOpenBaseConfig):
    system = KnownSystem.H200_SXM_141GBx1
    single_stream_expected_latency_ns = 1876267940
    model_path = "build/open-models/LLAMA2-70B-DepthPrunedMedusa/"
    quant_model_path = "INT4AWQ_quantized"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP, ModelOpt.DepthPrunedMedusa)
class H200_SXM_80GBx1_HighAccuracy(H200_SXM_141GBx1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, ModelOpt.DepthPrunedMedusa)
class Orin(SingleStreamGPUOpenBaseConfig):
    system = KnownSystem.Orin
    model_path = "build/open-models/LLAMA2-70B-DepthPrunedMedusa/"
    quant_model_path = "INT4AWQ_quantized"
    single_stream_expected_latency_ns = 5000000000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP, ModelOpt.DepthPrunedMedusa)
class Orin_HighAccuracy(Orin):
    pass
