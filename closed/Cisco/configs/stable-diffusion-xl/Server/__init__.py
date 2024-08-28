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
GPUBaseConfig = import_module("configs.stable-diffusion-xl").GPUBaseConfig


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server
    precision = "fp8"

    use_graphs = False
    gpu_inference_streams = 1
    gpu_copy_streams = 1
    sdxl_batcher_time_limit = 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GH200_96GB_aarch64x1(ServerGPUBaseConfig):
    system = KnownSystem.GH200_96GB_ARMx1
    gpu_batch_size = {'clip1': 8 * 2, 'clip2': 8 * 2, 'unet': 8 * 2, 'vae': 8}
    server_target_qps = 2.1
    sdxl_batcher_time_limit = 3
    use_graphs = False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx1(ServerGPUBaseConfig):
    system = KnownSystem.H100_SXM_80GBx1
    gpu_batch_size = {'clip1': 8 * 2, 'clip2': 8 * 2, 'unet': 8 * 2, 'vae': 8}
    server_target_qps = 2
    sdxl_batcher_time_limit = 3
    use_graphs = False  # disable to meet latency constraint for x1
    vboost_slider = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class C245M8_H100_PCIe_80GBx2(ServerGPUBaseConfig):
    system = KnownSystem.C245M8_H100_PCIe_80GBx2
    gpu_batch_size = {'clip1': 8 * 2, 'clip2': 8 * 2, 'unet': 8 * 2, 'vae': 8}
    server_target_qps = 2.3
    sdxl_batcher_time_limit = 3
    use_graphs = False

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class C245M8_L40Sx2(ServerGPUBaseConfig):
    system = KnownSystem.C245M8_L40Sx2
    precision = "int8"
    gpu_batch_size = {'clip1': 1 * 2, 'clip2': 1 * 2, 'unet': 1 * 2, 'vae': 1}
    server_target_qps = 1.27
    sdxl_batcher_time_limit = 0
    use_graphs = False
    min_query_count = 6 * 800

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class C240M7_L40Sx2(ServerGPUBaseConfig):
    system = KnownSystem.C240M7_L40Sx2
    precision = "int8"
    gpu_batch_size = {'clip1': 1 * 2, 'clip2': 1 * 2, 'unet': 1 * 2, 'vae': 1}
    server_target_qps = 1.28
    sdxl_batcher_time_limit = 0
    use_graphs = False
    min_query_count = 6 * 800

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class X210c_L40SX2(ServerGPUBaseConfig):
    system = KnownSystem.X210c_L40SX2
    precision = "int8"
    gpu_batch_size = {'clip1': 1 * 2, 'clip2': 1 * 2, 'unet': 1 * 2, 'vae': 1}
    server_target_qps = 1.27
    sdxl_batcher_time_limit = 0
    use_graphs = False
    min_query_count = 6 * 800

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx8(H100_SXM_80GBx1):
    system = KnownSystem.H100_SXM_80GBx8
    server_target_qps = 16.3
    sdxl_batcher_time_limit = 5
    use_graphs = False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class H100_SXM_80GBx8_MaxQ(H100_SXM_80GBx8):
    server_target_qps = 11
    power_limit = 350


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx1(ServerGPUBaseConfig):
    system = KnownSystem.H200_SXM_141GBx1
    gpu_batch_size = {'clip1': 8 * 2, 'clip2': 8 * 2, 'unet': 8 * 2, 'vae': 8}
    server_target_qps = 2.1
    sdxl_batcher_time_limit = 3
    use_graphs = False
    vboost_slider = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx8(H200_SXM_141GBx1):
    system = KnownSystem.H200_SXM_141GBx8
    server_target_qps = 16.9
    sdxl_batcher_time_limit = 5
    use_graphs = False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40Sx1(ServerGPUBaseConfig):
    system = KnownSystem.L40Sx1
    precision = "int8"
    gpu_batch_size = {'clip1': 1 * 2, 'clip2': 1 * 2, 'unet': 1 * 2, 'vae': 1}
    server_target_qps = 0.55
    sdxl_batcher_time_limit = 0
    use_graphs = False
    min_query_count = 6 * 800


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40Sx8(L40Sx1):
    system = KnownSystem.L40Sx8
    gpu_batch_size = {'clip1': 1 * 2, 'clip2': 1 * 2, 'unet': 1 * 2, 'vae': 1}
    sdxl_batcher_time_limit = 0
    server_target_qps = 5.05
    use_graphs = False
    min_query_count = 8 * 800
