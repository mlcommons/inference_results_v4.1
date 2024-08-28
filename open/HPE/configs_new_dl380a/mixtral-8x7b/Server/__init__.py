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

ParentConfig = import_module("configs.mixtral-8x7b")
GPUBaseConfig = ParentConfig.GPUBaseConfig


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    tensor_parallelism = 2
    pipeline_parallelism = 1
    precision = "fp16"
    enable_sort = False
    kvcache_free_gpu_mem_frac = 0.90
    min_duration = 2400000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GH200_144GB_aarch64x1(ServerGPUBaseConfig):
    system = KnownSystem.GH200_144GB_ARMx1
    gpu_batch_size = {'mixtral-8x7b': 1024}
    use_fp8 = True
    enable_sort = False
    tensor_parallelism = 1
    server_target_qps = 40


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GB_TP2x1(ServerGPUBaseConfig):
    system = KnownSystem.H100_SXM_80GBx2
    gpu_batch_size = {'mixtral-8x7b': 1024}
    use_fp8 = True
    tensor_parallelism = 2
    server_target_qps = 18.4
    vboost_slider = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GB_TP2x2(H100_SXM_80GB_TP2x1):
    system = KnownSystem.H100_SXM_80GBx4
    server_target_qps = 18.4 * 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GB_TP2x4(H100_SXM_80GB_TP2x2):
    system = KnownSystem.H100_SXM_80GBx8
    server_target_qps = 17.8 * 4


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class H100_SXM_80GB_TP2x4_MaxQ(H100_SXM_80GB_TP2x4):
    system = KnownSystem.H100_SXM_80GBx8
    server_target_qps = 13.5 * 4
    power_limit = 450


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx1(ServerGPUBaseConfig):
    system = KnownSystem.H200_SXM_141GBx1
    gpu_batch_size = {'mixtral-8x7b': 896}
    use_fp8 = True
    server_target_qps = 35
    enable_sort = False
    tensor_parallelism = 1
    vboost_slider = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx8(H200_SXM_141GBx1):
    system = KnownSystem.H200_SXM_141GBx8
    server_target_qps = 35 * 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GB_CTSx1(ServerGPUBaseConfig):
    system = KnownSystem.H200_SXM_141GB_CTSx1
    gpu_batch_size = {'mixtral-8x7b': 1024}
    use_fp8 = True
    server_target_qps = 38
    enable_sort = False
    tensor_parallelism = 1
    vboost_slider = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GB_CTSx8(H200_SXM_141GB_CTSx1):
    system = KnownSystem.H200_SXM_141GB_CTSx8
    server_target_qps = 38 * 8
