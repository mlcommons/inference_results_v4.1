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

ParentConfig = import_module("configs.dlrm-v2")
GPUBaseConfig = ParentConfig.GPUBaseConfig


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GH200_96GB_aarch64x1(ServerGPUBaseConfig):
    system = KnownSystem.GH200_96GB_ARMx1
    gpu_batch_size = {'dlrm-v2': 204800}
    embedding_weights_on_gpu_part: float = 1.0
    server_target_qps = 77500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class GH200_96GB_aarch64x1_High_Accuracy(GH200_96GB_aarch64x1):
    server_target_qps = 46200
    interaction_op_precision = 'fp16'


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class D54U_3U_H100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.D54U_3U_H100_PCIe_80GBx4
    gpu_batch_size = {'dlrm-v2': 51200}
    embedding_weights_on_gpu_part: float = 1.0
    server_target_qps = 175000
    server_num_issue_query_threads = 4
    numa_config = "0,1:0-51,104-155&2,3:52-103,156-207"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class D54U_3U_H100_PCIe_80GBx4_High_Accuracy(D54U_3U_H100_PCIe_80GBx4):
    server_target_qps = 100000
    interaction_op_precision = 'fp16'


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class D54U_3U_L40S_PCIe_48GBx4(ServerGPUBaseConfig):
    system = KnownSystem.D54U_3U_L40S_PCIe_48GBx4
    gpu_batch_size = {'dlrm-v2': 6000}
    embedding_weights_on_gpu_part: float = 1.0
    server_target_qps = 84400
    numa_config = "0,1:0-43,88-131&2,3:44-87,132-175"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class D54U_3U_L40S_PCIe_48GBx4_HighAccuracy(D54U_3U_L40S_PCIe_48GBx4):
    server_target_qps = 51000
    interaction_op_precision = 'fp16'
    top_mlp_precision = 'fp16'



