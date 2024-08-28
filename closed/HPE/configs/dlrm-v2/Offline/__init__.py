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


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline
    check_contiguity = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GH200_96GB_aarch64x1(OfflineGPUBaseConfig):
    system = KnownSystem.GH200_96GB_ARMx1
    gpu_batch_size = {'dlrm-v2': 600_000}
    embedding_weights_on_gpu_part: float = 1.0
    offline_expected_qps = 87000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class GH200_96GB_aarch64x1_High_Accuracy(GH200_96GB_aarch64x1):
    offline_expected_qps = 53000
    interaction_op_precision = 'fp16'


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GH200_144GB_aarch64x1(OfflineGPUBaseConfig):
    system = KnownSystem.GH200_144GB_ARMx1
    gpu_batch_size = {'dlrm-v2': 600_000}
    embedding_weights_on_gpu_part: float = 1.0
    offline_expected_qps = 99000
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class GH200_144GB_aarch64x1_High_Accuracy(GH200_144GB_aarch64x1):
    offline_expected_qps = 56700
    interaction_op_precision = 'fp16'


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GH200_144GB_aarch64x2(GH200_144GB_aarch64x1):
    system = KnownSystem.GH200_144GB_ARMx2
    offline_expected_qps = GH200_144GB_aarch64x1.offline_expected_qps * 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class GH200_144GB_aarch64x2_HighAccuracy(GH200_144GB_aarch64x2):
    offline_expected_qps = GH200_144GB_aarch64x1_High_Accuracy.offline_expected_qps * 2
    interaction_op_precision = 'fp16'


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.H100_SXM_80GBx1
    gpu_batch_size = {'dlrm-v2': 600_000}
    embedding_weights_on_gpu_part: float = 1.0
    offline_expected_qps = 77000
    vboost_slider = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_SXM_80GBx1_HighAccuracy(H100_SXM_80GBx1):
    offline_expected_qps = 47500
    interaction_op_precision = 'fp16'


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx8(H100_SXM_80GBx1):
    system = KnownSystem.H100_SXM_80GBx8
    offline_expected_qps = H100_SXM_80GBx1.offline_expected_qps * 8
    numa_config = "0-3:0-55,112-167&4-7:56-111,168-223"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_SXM_80GBx8_HighAccuracy(H100_SXM_80GBx8):
    system = KnownSystem.H100_SXM_80GBx8
    offline_expected_qps = H100_SXM_80GBx1_HighAccuracy.offline_expected_qps * 8
    interaction_op_precision = 'fp16'


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class H100_SXM_80GBx8_MaxQ(H100_SXM_80GBx8):
    offline_expected_qps = 56800 * 8
    power_limit = 450


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class H100_SXM_80GBx8_HighAccuracy_MaxQ(H100_SXM_80GBx8_MaxQ):
    offline_expected_qps = 35000 * 8
    interaction_op_precision = 'fp16'


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.H200_SXM_141GBx1
    gpu_batch_size = {'dlrm-v2': 600_000}
    embedding_weights_on_gpu_part: float = 1.0
    offline_expected_qps = 90000
    start_from_device = True
    vboost_slider = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H200_SXM_141GBx1_HighAccuracy(H200_SXM_141GBx1):
    offline_expected_qps = 51000
    interaction_op_precision = 'fp16'


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx8(H200_SXM_141GBx1):
    system = KnownSystem.H200_SXM_141GBx8
    offline_expected_qps = H200_SXM_141GBx1.offline_expected_qps * 8
    # numa_config = "0-3:0-55,112-167&4-7:56-111,168-223"

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class H200_SXM_141GBx8_MaxQ(H200_SXM_141GBx8):
    offline_expected_qps = 58000 * 8
    power_limit = 500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H200_SXM_141GBx8_HighAccuracy(H200_SXM_141GBx8):
    system = KnownSystem.H200_SXM_141GBx8
    offline_expected_qps = H200_SXM_141GBx1_HighAccuracy.offline_expected_qps * 8
    interaction_op_precision = 'fp16'

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class H200_SXM_141GBx8_HighAccuracy_MaxQ(H200_SXM_141GBx8_HighAccuracy):
    offline_expected_qps = 39000 * 8
    power_limit = 500



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GB_CTSx1(OfflineGPUBaseConfig):
    system = KnownSystem.H200_SXM_141GB_CTSx1
    gpu_batch_size = {'dlrm-v2': 600_000}
    embedding_weights_on_gpu_part: float = 1.0
    offline_expected_qps = 100000
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H200_SXM_141GB_CTSx1_HighAccuracy(H200_SXM_141GB_CTSx1):
    offline_expected_qps = 60000
    interaction_op_precision = 'fp16'


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GB_CTSx8(H200_SXM_141GB_CTSx1):
    system = KnownSystem.H200_SXM_141GB_CTSx8
    offline_expected_qps = H200_SXM_141GB_CTSx1.offline_expected_qps * 8
    # numa_config = "0-3:0-55,112-167&4-7:56-111,168-223"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H200_SXM_141GB_CTSx8_HighAccuracy(H200_SXM_141GB_CTSx8):
    system = KnownSystem.H200_SXM_141GB_CTSx8
    offline_expected_qps = H200_SXM_141GB_CTSx1_HighAccuracy.offline_expected_qps * 8
    interaction_op_precision = 'fp16'


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.H100_PCIe_80GBx1
    gpu_batch_size = {'dlrm-v2': 51200}
    embedding_weights_on_gpu_part: float = 1.0
    offline_expected_qps = 23800


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_PCIe_80GBx1_HighAccuracy(H100_PCIe_80GBx1):
    interaction_op_precision = 'fp16'


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GBx8(H100_PCIe_80GBx1):
    system = KnownSystem.H100_PCIe_80GBx8
    offline_expected_qps = 193000
    numa_config = "3:0-15,128-143&2:16-31,144-159&1:32-47,160-175&0:48-63,176-191&7:64-79,192-207&6:80-95,208-223&5:96-111,224-239&4:112-127,240-255"
    qsl_numa_override = "0-3&4-7"
    use_batcher_thread_per_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_PCIe_80GBx8_HighAccuracy(H100_PCIe_80GBx8):
    system = KnownSystem.H100_PCIe_80GBx8
    offline_expected_qps = H100_PCIe_80GBx1_HighAccuracy.offline_expected_qps * 8
    interaction_op_precision = 'fp16'


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class H100_PCIe_80GBx8_MaxQ(H100_PCIe_80GBx8):
    offline_expected_qps = 20100 * 8
    use_batcher_thread_per_device = False
    power_limit = 275


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class H100_PCIe_80GBx8_HighAccuracy_MaxQ(H100_PCIe_80GBx8_MaxQ):
    interaction_op_precision = 'fp16'


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_NVL_94GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.H100_NVL_94GBx1
    gpu_batch_size = {'dlrm-v2': 51200}
    embedding_weights_on_gpu_part: float = 1.0
    offline_expected_qps = 23800


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_NVL_94GBx1_HighAccuracy(H100_NVL_94GBx1):
    interaction_op_precision = 'fp16'


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_NVL_94GBx8(H100_NVL_94GBx1):
    system = KnownSystem.H100_NVL_94GBx8
    offline_expected_qps = 193000
    numa_config = "3:0-15,128-143&2:16-31,144-159&1:32-47,160-175&0:48-63,176-191&7:64-79,192-207&6:80-95,208-223&5:96-111,224-239&4:112-127,240-255"
    qsl_numa_override = "0-3&4-7"
    use_batcher_thread_per_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_NVL_94GBx8_HighAccuracy(H100_NVL_94GBx8):
    system = KnownSystem.H100_NVL_94GBx8
    offline_expected_qps = H100_NVL_94GBx1_HighAccuracy.offline_expected_qps * 8
    interaction_op_precision = 'fp16'


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L4x1(OfflineGPUBaseConfig):
    system = KnownSystem.L4x1
    embedding_weights_on_gpu_part = 0.30
    gpu_batch_size = {'dlrm-v2': 1400}
    offline_expected_qps = 3400


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L4x1_HighAccuracy(L4x1):
    interaction_op_precision = 'fp16'


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40Sx1(OfflineGPUBaseConfig):
    system = KnownSystem.L40Sx1
    embedding_weights_on_gpu_part: float = 1.0
    gpu_batch_size = {'dlrm-v2': 7500}
    offline_expected_qps = 25000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L40Sx1_HighAccuracy(L40Sx1):
    offline_expected_qps = 12700
    interaction_op_precision = 'fp16'


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40Sx8(L40Sx1):
    system = KnownSystem.L40Sx8
    offline_expected_qps = 180000
    numa_config = "3:0-15,128-143&2:16-31,144-159&1:32-47,160-175&0:48-63,176-191&7:64-79,192-207&6:80-95,208-223&5:96-111,224-239&4:112-127,240-255"
    qsl_numa_override = "0-1&2-3&4-5&6-7"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L40Sx8_HighAccuracy(L40Sx8):
    offline_expected_qps = 93000
    interaction_op_precision = 'fp16'
    top_mlp_precision = 'fp16'
