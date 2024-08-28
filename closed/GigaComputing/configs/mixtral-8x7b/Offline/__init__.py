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


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    tensor_parallelism = 1
    pipeline_parallelism = 1
    precision = "fp16"
    kvcache_free_gpu_mem_frac = 0.90
    min_duration = 2400000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GH200_144GB_aarch64x1(OfflineGPUBaseConfig):
    system = KnownSystem.GH200_144GB_ARMx1
    gpu_batch_size = {'mixtral-8x7b': 806}
    use_fp8 = True
    offline_expected_qps = 45
    tensor_parallelism = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.H100_SXM_80GBx1
    gpu_batch_size = {'mixtral-8x7b': 896}
    use_fp8 = True
    offline_expected_qps = 46
    enable_sort = False
    tensor_parallelism = 1
    max_num_tokens = 8192
    vboost_slider = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GB_TP2x1(OfflineGPUBaseConfig):
    system = KnownSystem.H100_SXM_80GBx2
    gpu_batch_size = {'mixtral-8x7b': 4096}
    use_fp8 = True
    offline_expected_qps = 95
    enable_sort = False
    tensor_parallelism = 2
    vboost_slider = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx8(H100_SXM_80GBx1):
    system = KnownSystem.H100_SXM_80GBx8
    offline_expected_qps = 46 * 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.H200_SXM_141GBx1
    gpu_batch_size = {'mixtral-8x7b': 3072}
    use_fp8 = True
    offline_expected_qps = 53
    enable_sort = False
    tensor_parallelism = 1
    vboost_slider = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx8(H200_SXM_141GBx1):
    system = KnownSystem.H200_SXM_141GBx8
    offline_expected_qps = 53 * 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class H200_SXM_141GBx8_MaxQ(H200_SXM_141GBx8):
    power_limit = 500
    offline_expected_qps = 41 * 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GB_CTSx1(OfflineGPUBaseConfig):
    system = KnownSystem.H200_SXM_141GB_CTSx1
    gpu_batch_size = {'mixtral-8x7b': 3072}
    use_fp8 = True
    offline_expected_qps = 55
    enable_sort = False
    tensor_parallelism = 1
    vboost_slider = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GB_CTSx8(H200_SXM_141GB_CTSx1):
    system = KnownSystem.H200_SXM_141GB_CTSx8
    offline_expected_qps = 450
