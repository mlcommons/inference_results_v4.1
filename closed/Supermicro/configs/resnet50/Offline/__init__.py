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


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline

    run_infer_on_copy_streams = False
    use_graphs = False
    gpu_inference_streams = 1
    gpu_copy_streams = 2


class GH200_96GB_aarch64x1(OfflineGPUBaseConfig):
    gpu_batch_size = {'resnet50': 2048}
    offline_expected_qps = 90000
    start_from_device = True


class L4x1(OfflineGPUBaseConfig):
    gpu_batch_size = {'resnet50': 32}
    gpu_inference_streams = 1
    gpu_copy_streams = 2
    offline_expected_qps = 13000
    use_graphs = True


class L40x1(OfflineGPUBaseConfig):
    gpu_batch_size = {'resnet50': 2048}
    offline_expected_qps = 40000


class L40Sx1(OfflineGPUBaseConfig):
    gpu_batch_size = {'resnet50': 64}
    gpu_inference_streams = 1
    gpu_copy_streams = 2
    offline_expected_qps = 40000


class H100_SXM_80GBx1(OfflineGPUBaseConfig):
    gpu_batch_size = {'resnet50': 2048}
    offline_expected_qps = 90000
    start_from_device = True


class H100_SXM_80GBx1_Triton(H100_SXM_80GBx1):
    use_triton = True


class H100_SXM_80GBx8(H100_SXM_80GBx1):
    offline_expected_qps = 90000 * 8


class H100_SXM_80GBx8_MaxQ(H100_SXM_80GBx8):
    power_limit = 300
    offline_expected_qps = 480000


class H100_SXM_80GBx8_Triton(H100_SXM_80GBx8):
    use_triton = True


class H200_SXM_141GBx1(OfflineGPUBaseConfig):
    gpu_batch_size = {'resnet50': 2048}
    offline_expected_qps = 90000
    start_from_device = True


class H200_SXM_141GBx8(H200_SXM_141GBx1):
    offline_expected_qps = 90000 * 8


class H200_SXM_141GB_CTSx1(OfflineGPUBaseConfig):
    gpu_batch_size = {'resnet50': 2048}
    offline_expected_qps = 90000
    start_from_device = True


class H200_SXM_141GB_CTSx8(H200_SXM_141GB_CTSx1):
    pass

class H100_PCIe_80GBx1(OfflineGPUBaseConfig):
    gpu_batch_size = {'resnet50': 2048}
    offline_expected_qps = 57000


class H100_NVL_94GBx1(OfflineGPUBaseConfig):
    gpu_batch_size = {'resnet50': 2048}
    offline_expected_qps = 57000


class H100_PCIe_80GBx1_Triton(H100_PCIe_80GBx1):
    use_triton = True


class H100_PCIe_80GBx8(H100_PCIe_80GBx1):
    offline_expected_qps = 450000
    numa_config = "3:0-15&2:16-31&1:32-47&0:48-63&7:64-79&6:80-95&5:96-111&4:112-127"
    # numa_config = "3:0-7&2:8-15&1:16-23&0:24-31&7:32-39&6:40-47&5:48-55&4:56-63"    $ for ipp1-1470


class H100_NVL_94GBx8(H100_NVL_94GBx1):
    offline_expected_qps = 450000
    numa_config = "3:0-15&2:16-31&1:32-47&0:48-63&7:64-79&6:80-95&5:96-111&4:112-127"
    # numa_config = "3:0-7&2:8-15&1:16-23&0:24-31&7:32-39&6:40-47&5:48-55&4:56-63"    $ for ipp1-1470


class H100_PCIe_80GBx8_Triton(H100_PCIe_80GBx8):
    use_triton = True


class H100_PCIe_80GB_aarch64x1(OfflineGPUBaseConfig):
    gpu_batch_size = {'resnet50': 2048}
    offline_expected_qps = 52000


class H100_PCIe_80GB_aarch64x4(H100_PCIe_80GB_aarch64x1):
    offline_expected_qps = 200000


class A100_PCIe_80GBx1(OfflineGPUBaseConfig):
    gpu_batch_size = {'resnet50': 2048}
    offline_expected_qps = 40000


class A100_PCIe_80GBx1_Triton(A100_PCIe_80GBx1):
    use_triton = True


class A100_PCIe_80GBx1_TritonUnified(A100_PCIe_80GBx1):
    use_triton = True
    batch_triton_requests = True


class A100_PCIe_80GBx8(A100_PCIe_80GBx1):
    offline_expected_qps = 293000.0
    numa_config = "3:0-15,128-143&2:16-31,144-159&1:32-47,160-175&0:48-63,176-191&7:64-79,192-207&6:80-95,208-223&5:96-111,224-239&4:112-127,240-255"


class A100_PCIe_80GBx8_Triton(A100_PCIe_80GBx8):
    use_triton = True
    num_concurrent_batchers = 6
    num_concurrent_issuers = 6
    batch_triton_requests = True


class A100_PCIe_80GBx8_TritonUnified(A100_PCIe_80GBx8):
    use_triton = True
    num_concurrent_batchers = 6
    num_concurrent_issuers = 6
    batch_triton_requests = True


class A100_PCIe_80GBx8_MaxQ(A100_PCIe_80GBx8):
    power_limit = 195
    offline_expected_qps = 250000
    numa_config = None


class A100_PCIe_80GBx8_Triton_MaxQ(A100_PCIe_80GBx8_MaxQ):
    use_triton = True
    offline_expected_qps = 256000.0


class A100_PCIe_80GBx8_TritonUnified_MaxQ(A100_PCIe_80GBx8_MaxQ):
    use_triton = True
    offline_expected_qps = 256000.0
    batch_triton_requests = True


class A100_PCIe_80GB_aarch64x1(OfflineGPUBaseConfig):
    gpu_batch_size = {'resnet50': 2048}
    offline_expected_qps = 36000


class A100_PCIe_80GB_aarch64x1_Triton(A100_PCIe_80GB_aarch64x1):
    use_triton = True


class A100_PCIe_80GB_aarch64x1_TritonUnified(A100_PCIe_80GB_aarch64x1):
    use_triton = True
    batch_triton_requests = True


class A100_PCIe_80GB_aarch64x2(A100_PCIe_80GB_aarch64x1):
    offline_expected_qps = 72000.0


class A100_PCIe_80GB_aarch64x2_Triton(A100_PCIe_80GB_aarch64x2):
    use_triton = True


class A100_PCIe_80GB_aarch64x2_TritonUnified(A100_PCIe_80GB_aarch64x2):
    use_triton = True
    batch_triton_requests = True


class A100_PCIe_80GB_aarch64x4(A100_PCIe_80GB_aarch64x1):
    offline_expected_qps = 140000.0


class A100_PCIe_80GB_aarch64x4_Triton(A100_PCIe_80GB_aarch64x4):
    use_triton = True


class A100_PCIe_80GB_aarch64x4_TritonUnified(A100_PCIe_80GB_aarch64x4):
    use_triton = True
    batch_triton_requests = True


class A100_PCIe_80GB_aarch64x4_MaxQ(A100_PCIe_80GB_aarch64x4):
    offline_expected_qps = 120000.0
    power_limit = 175


class A100_PCIe_aarch64x1(OfflineGPUBaseConfig):
    gpu_batch_size = {'resnet50': 2048}
    offline_expected_qps = 36000


class A100_PCIe_aarch64x1_Triton(A100_PCIe_aarch64x1):
    use_triton = True


class A100_PCIe_aarch64x1_TritonUnified(A100_PCIe_aarch64x1):
    use_triton = True
    batch_triton_requests = True


class A100_PCIe_aarch64x2(A100_PCIe_aarch64x1):
    offline_expected_qps = 72000.0


class A100_PCIe_aarch64x2_Triton(A100_PCIe_aarch64x2):
    use_triton = True


class A100_PCIe_aarch64x2_TritonUnified(A100_PCIe_aarch64x2):
    use_triton = True
    batch_triton_requests = True


class A100_PCIe_aarch64x4(A100_PCIe_aarch64x1):
    offline_expected_qps = 140000.0


class A100_PCIe_aarch64x4_Triton(A100_PCIe_aarch64x4):
    use_triton = True


class A100_PCIe_aarch64x4_TritonUnified(A100_PCIe_aarch64x4):
    use_triton = True
    batch_triton_requests = True


class A100_PCIe_aarch64x4_MaxQ(A100_PCIe_aarch64x4):
    offline_expected_qps = 120000.0
    power_limit = 175


class A100_PCIe_MIG_1x1g5gb(OfflineGPUBaseConfig):
    gpu_batch_size = {'resnet50': 256}
    run_infer_on_copy_streams = True
    offline_expected_qps = 5100


class A100_PCIe_MIG_1x1g5gb_Triton(A100_PCIe_MIG_1x1g5gb):
    use_triton = True


class A100_PCIe_MIG_1x1g5gb_TritonUnified(A100_PCIe_MIG_1x1g5gb):
    use_triton = True
    batch_triton_requests = True


class A100_PCIe_80GB_MIG_1x1g10gb(OfflineGPUBaseConfig):
    gpu_batch_size = {'resnet50': 256}
    run_infer_on_copy_streams = True
    offline_expected_qps = 4800


class A100_PCIe_80GB_MIG_1x1g10gb_Hetero(A100_PCIe_80GB_MIG_1x1g10gb):
    pass


class A100_PCIe_80GB_MIG_1x1g10gb_Triton(A100_PCIe_80GB_MIG_1x1g10gb):
    use_triton = True


class A100_PCIe_80GB_MIG_1x1g10gb_TritonUnified(A100_PCIe_80GB_MIG_1x1g10gb):
    use_triton = True
    batch_triton_requests = True


class A100_SXM_80GB_MIG_1x1g10gb(OfflineGPUBaseConfig):
    gpu_batch_size = {'resnet50': 256}
    run_infer_on_copy_streams = True
    start_from_device = True
    offline_expected_qps = 5100


class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    pass


class A100_SXM_80GB_MIG_1x1g10gb_Triton(A100_SXM_80GB_MIG_1x1g10gb):
    use_triton = True


class A100_SXM_80GB_MIG_1x1g10gb_TritonUnified(A100_SXM_80GB_MIG_1x1g10gb):
    use_triton = True
    batch_triton_requests = True


class A100_SXM_80GBx1(OfflineGPUBaseConfig):
    gpu_batch_size = {'resnet50': 2048}
    gpu_copy_streams = 4
    run_infer_on_copy_streams = True
    start_from_device = True
    offline_expected_qps = 43000


class A100_SXM_80GBx1_Triton(A100_SXM_80GBx1):
    start_from_device = None
    gpu_copy_streams = 1
    use_triton = True


class A100_SXM_80GBx1_TritonUnified(A100_SXM_80GBx1):
    start_from_device = None
    gpu_copy_streams = 1
    use_triton = True
    batch_triton_requests = True


class A100_SXM_80GBx8(A100_SXM_80GBx1):
    run_infer_on_copy_streams = False
    gpu_inference_streams = 2
    gpu_copy_streams = 3
    offline_expected_qps = 340000


class A100_SXM_80GBx8_Triton(A100_SXM_80GBx8):
    gpu_copy_streams = 1
    num_concurrent_batchers = 6
    num_concurrent_issuers = 6
    instance_group_count = 2
    batch_triton_requests = True
    use_triton = True
    offline_expected_qps = 340000


class A100_SXM_80GBx8_TritonUnified(A100_SXM_80GBx8):
    gpu_copy_streams = 1
    num_concurrent_batchers = 6
    num_concurrent_issuers = 6
    instance_group_count = 2
    use_triton = True
    offline_expected_qps = 340000
    batch_triton_requests = True


class A100_SXM_80GBx8_MaxQ(A100_SXM_80GBx8):
    power_limit = 250


class A100_SXM_80GBx8_Triton_MaxQ(A100_SXM_80GBx8_MaxQ):
    start_from_device = None
    gpu_copy_streams = 1
    use_triton = True
    offline_expected_qps = 270000


class A100_SXM_80GBx8_TritonUnified_MaxQ(A100_SXM_80GBx8_MaxQ):
    start_from_device = None
    gpu_copy_streams = 1
    use_triton = True
    offline_expected_qps = 270000
    batch_triton_requests = True


class A100_SXM_80GB_aarch64x1(OfflineGPUBaseConfig):
    gpu_batch_size = {'resnet50': 2048}
    offline_expected_qps = 39500


class A100_SXM_80GB_aarch64x1_Triton(A100_SXM_80GB_aarch64x1):
    use_triton = True


class A100_SXM_80GB_aarch64x1_TritonUnified(A100_SXM_80GB_aarch64x1):
    use_triton = True
    batch_triton_requests = True


class A100_SXM_80GB_aarch64x8(A100_SXM_80GB_aarch64x1):
    offline_expected_qps = 300000


class A100_SXM_80GB_aarch64x8_MaxQ(A100_SXM_80GB_aarch64x8):
    offline_expected_qps = 250000
    power_limit = 250


class A100_SXM_80GB_aarch64x8_Triton(A100_SXM_80GB_aarch64x8):
    use_triton = True
    offline_expected_qps = 300000


class A100_SXM_80GB_aarch64x8_TritonUnified(A100_SXM_80GB_aarch64x8):
    use_triton = True
    offline_expected_qps = 300000
    batch_triton_requests = True


class A100_SXM_80GB_aarch64_MIG_1x1g10gb(OfflineGPUBaseConfig):
    gpu_batch_size = {'resnet50': 256}
    run_infer_on_copy_streams = True
    offline_expected_qps = 5100


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Hetero(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    pass


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Triton(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    use_triton = True


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_TritonUnified(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    use_triton = True
    batch_triton_requests = True


class A100_SXM4_40GB_MIG_1x1g5gb(OfflineGPUBaseConfig):
    gpu_batch_size = {'resnet50': 256}
    run_infer_on_copy_streams = True
    start_from_device = True
    offline_expected_qps = 5100


class A100_SXM4_40GB_MIG_1x1g5gb_Triton(A100_SXM4_40GB_MIG_1x1g5gb):
    use_triton = True


class A100_SXM4_40GB_MIG_1x1g5gb_TritonUnified(A100_SXM4_40GB_MIG_1x1g5gb):
    use_triton = True
    batch_triton_requests = True


class A100_SXM4_40GBx1(OfflineGPUBaseConfig):
    gpu_batch_size = {'resnet50': 2048}
    gpu_copy_streams = 4
    run_infer_on_copy_streams = True
    start_from_device = True
    offline_expected_qps = 36800


class A100_SXM4_40GBx1_Triton(A100_SXM4_40GBx1):
    start_from_device = None
    gpu_copy_streams = 1
    use_triton = True


class A100_SXM4_40GBx1_TritonUnified(A100_SXM4_40GBx1):
    start_from_device = None
    gpu_copy_streams = 1
    use_triton = True
    batch_triton_requests = True


class A100_SXM4_40GBx8(A100_SXM4_40GBx1):
    run_infer_on_copy_streams = False
    gpu_inference_streams = 2
    gpu_copy_streams = 3
    offline_expected_qps = 294400


class A100_SXM4_40GBx8_Triton(A100_SXM4_40GBx8):
    start_from_device = None
    gpu_copy_streams = 1
    use_triton = True


class A100_SXM4_40GBx8_TritonUnified(A100_SXM4_40GBx8):
    start_from_device = None
    gpu_copy_streams = 1
    use_triton = True
    batch_triton_requests = True


class Orin(OfflineGPUBaseConfig):
    # GPU-only QPS
    _gpu_offline_expected_qps = 4500
    # DLA-only (1x) QPS
    _dla_offline_expected_qps = 600
    # GPU + 2 DLA QPS
    offline_expected_qps = 6100
    use_direct_host_access = True
    run_infer_on_copy_streams = None
    dla_batch_size = {'backbone': 2, 'topk': 2 * 4}
    dla_copy_streams = 2
    dla_inference_streams = 1
    dla_core = 0
    gpu_batch_size = {'resnet50': 256}
    input_format = 'chw4'
    tensor_path = 'build/preprocessed_data/imagenet/ResNet50/int8_chw4/'


class Orin_Triton(Orin):
    use_triton = True
    batch_triton_requests = True


class Orin_TritonUnified(Orin):
    use_triton = True
    batch_triton_requests = True


class Orin_MaxQ(Orin):
    # NOTE: Orin AGX 3.1 Shmoo
    soc_cpu_freq = 576000
    soc_gpu_freq = 612000000
    soc_dla_freq = 780200000
    soc_emc_freq = 2133000000
    soc_pva_freq = 115000000
    orin_num_cores = 4
    offline_expected_qps = 3400
    gpu_batch_size = {'resnet50': 256}
    dla_batch_size = {'resnet50': 8}


class Orin_NX(OfflineGPUBaseConfig):
    # GPU-only QPS
    _gpu_offline_expected_qps = 1700
    # DLA-only (1x) QPS
    _dla_offline_expected_qps = 230
    # GPU + 2 DLA QPS
    offline_expected_qps = 2600
    use_direct_host_access = True
    run_infer_on_copy_streams = None
    dla_batch_size = {'backbone': 2, 'topk': 2 * 4}
    dla_copy_streams = 2
    dla_inference_streams = 1
    dla_core = 0
    gpu_batch_size = {'resnet50': 256}
    input_format = 'chw4'
    tensor_path = 'build/preprocessed_data/imagenet/ResNet50/int8_chw4/'


class Orin_NX_MaxQ(Orin_NX):
    # NOTE: Orin NX 3.1 Shmoo
    soc_cpu_freq = 652800
    soc_gpu_freq = 510000000
    soc_dla_freq = 1164200000
    soc_emc_freq = 2133000000
    soc_pva_freq = 0
    orin_num_cores = 4
    offline_expected_qps = 1700
    gpu_batch_size = {'resnet50': 256}
    dla_batch_size = {'resnet50': 8}


class H100_PCIe_80GBx8_MaxQ(H100_PCIe_80GBx8):
    offline_expected_qps = 360000
    power_limit = 215
    energy_aware_kernels = True
    numa_config = "3:0-7&2:8-15&1:16-23&0:24-31&7:32-39&6:40-47&5:48-55&4:56-63"
