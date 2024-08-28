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


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server
    use_graphs = True
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    use_small_tile_gemm_plugin = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GH200_96GB_aarch64x1(ServerGPUBaseConfig):
    system = KnownSystem.GH200_96GB_ARMx1
    use_small_tile_gemm_plugin = False
    use_graphs = False
    gpu_batch_size = {'bert': 256}
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 7100
    server_num_issue_query_threads = 1
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class GH200_96GB_aarch64x1_High_Accuracy(GH200_96GB_aarch64x1):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = {'bert': 512}
    server_target_qps = 6600


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class D54U_3U_H100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.D54U_3U_H100_PCIe_80GBx4
    use_small_tile_gemm_plugin = False
    use_graphs = False
    graphs_max_seqlen = 200
    gpu_batch_size = {'bert': 256}
    server_target_qps = 17760
    server_num_issue_query_threads = 1
    workspace_size = 7516192768
    soft_drop = 1.0


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class D54U_3U_H100_PCIe_80GBx4_HighAccuracy(D54U_3U_H100_PCIe_80GBx4):
    precision = 'fp16'
    gpu_batch_size = {'bert': 256}
    use_fp8 = True
    server_target_qps = 16000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class D54U_3U_L40S_PCIe_48GBx4(ServerGPUBaseConfig):
    system = KnownSystem.D54U_3U_L40S_PCIe_48GBx4
    gpu_batch_size = {'bert': 64}
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 12000.0
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class D54U_3U_L40S_PCIe_48GBx4_HighAccuracy(D54U_3U_L40S_PCIe_48GBx4):
    precision = "fp16"
    server_target_qps = 5400



