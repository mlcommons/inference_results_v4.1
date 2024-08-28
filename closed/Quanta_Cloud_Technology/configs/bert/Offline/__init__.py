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

from code.common.constants import Benchmark, Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *
from configs.bert import GPUBaseConfig


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline

    gpu_copy_streams = 2
    gpu_inference_streams = 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GH200_96GB_aarch64x1(OfflineGPUBaseConfig):
    system = KnownSystem.GH200_96GB_ARMx1
    use_small_tile_gemm_plugin = False
    gpu_batch_size = {'bert': 1280}
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    offline_expected_qps = 10000
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class GH200_96GB_aarch64x1_High_Accuracy(GH200_96GB_aarch64x1):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = {'bert': 1024}
    offline_expected_qps = 8600


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class D54U_3U_H100_PCIe_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.D54U_3U_H100_PCIe_80GBx4
    use_small_tile_gemm_plugin = False
    gpu_batch_size = {'bert': 1280}
    offline_expected_qps = 22800
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class D54U_3U_H100_PCIe_80GBx4_HighAccuracy(D54U_3U_H100_PCIe_80GBx4):
    precision = "fp16"
    use_fp8 = True
    offline_expected_qps = 20000
    use_graphs = False
    gpu_batch_size = {'bert': 1024}


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class D54U_3U_L40S_PCIe_48GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.D54U_3U_L40S_PCIe_48GBx4
    use_small_tile_gemm_plugin = True
    gpu_batch_size = {'bert': 32}
    offline_expected_qps = 13600
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class D54U_3U_L40S_PCIe_48GBx4_HighAccuracy(OfflineGPUBaseConfig):
    system = KnownSystem.D54U_3U_L40S_PCIe_48GBx4
    precision = "fp16"
    use_fp8 = True
    offline_expected_qps = 7000
    gpu_batch_size = {'bert': 32}



