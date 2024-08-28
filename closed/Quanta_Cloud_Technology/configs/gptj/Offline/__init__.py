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
from configs.gptj import GPUBaseConfig


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    tensor_parallelism = 1
    precision = "fp16"
    enable_sort = False
    num_sort_segments = 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GH200_96GB_aarch64x1(OfflineGPUBaseConfig):
    system = KnownSystem.GH200_96GB_ARMx1
    gpu_batch_size = {'gptj': 480}
    use_fp8 = True
    enable_sort = False
    offline_expected_qps = 36 


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class GH200_96GB_aarch64x1_HighAccuracy(GH200_96GB_aarch64x1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class D54U_3U_L40S_PCIe_48GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.D54U_3U_L40S_PCIe_48GBx4
    gpu_batch_size = {'gptj': 100}
    use_fp8 = True
    offline_expected_qps = 55
    enable_sort = False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class D54U_3U_L40S_PCIe_48GBx4_HighAccuracy(D54U_3U_L40S_PCIe_48GBx4):
    pass



