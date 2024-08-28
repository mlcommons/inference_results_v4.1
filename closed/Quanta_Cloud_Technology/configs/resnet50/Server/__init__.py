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
from configs.resnet50 import GPUBaseConfig


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server
    active_sms = 100

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class GH200_96GB_aarch64x1(ServerGPUBaseConfig):
    system = KnownSystem.GH200_96GB_ARMx1
    use_deque_limit = True
    deque_timeout_usec = 3000
    gpu_batch_size = {'resnet50': 256}
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    server_target_qps = 77000
    use_cuda_thread_per_device = True
    use_graphs = True
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class D54U_3U_H100_PCIE_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.D54U_3U_H100_PCIe_80GBx4
    gpu_batch_size = {'resnet50': 128}
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 188000
    use_cuda_thread_per_device = True
    use_batcher_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class D54U_3U_L40S_PCIe_48GBx4(ServerGPUBaseConfig):
    system = KnownSystem.D54U_3U_L40S_PCIe_48GBx4
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = {'resnet50': 128}
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 150000
    use_cuda_thread_per_device = True
    use_graphs = True



