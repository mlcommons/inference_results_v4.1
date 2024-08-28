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


class GH200_96GB_aarch64x1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = False
    gpu_batch_size = {'bert': 1280}
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    offline_expected_qps = 10000
    workspace_size = 7516192768


class GH200_96GB_aarch64x1_High_Accuracy(GH200_96GB_aarch64x1):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = {'bert': 1024}
    offline_expected_qps = 8600


class H100_PCIe_80GBx1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = False
    gpu_batch_size = {'bert': 1280}
    offline_expected_qps = 5700
    workspace_size = 7516192768


class H100_PCIe_80GBx1_HighAccuracy(H100_PCIe_80GBx1):
    precision = "fp16"
    use_fp8 = True
    offline_expected_qps = 5000
    use_graphs = False
    gpu_batch_size = {'bert': 1024}


class H100_NVL_94GBx1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = False
    gpu_batch_size = {'bert': 1280}
    offline_expected_qps = 5700
    workspace_size = 7516192768


class H100_NVL_94GBx1_HighAccuracy(H100_NVL_94GBx1):
    precision = "fp16"
    use_fp8 = True
    offline_expected_qps = 5000
    use_graphs = False
    gpu_batch_size = {'bert': 1024}


class H100_PCIe_80GBx1_Triton(H100_PCIe_80GBx1):
    use_triton = True


class H100_PCIe_80GBx1_HighAccuracy_Triton(H100_PCIe_80GBx1_HighAccuracy):
    offline_expected_qps = 1800
    use_triton = True


class H100_PCIe_80GBx8(H100_PCIe_80GBx1):
    offline_expected_qps = 46000


class H100_PCIe_80GBx8_HighAccuracy(H100_PCIe_80GBx1_HighAccuracy):
    offline_expected_qps = 5000 * 8


class H100_NVL_94GBx8(H100_NVL_94GBx1):
    offline_expected_qps = 46000


class H100_NVL_94GBx8_HighAccuracy(H100_NVL_94GBx1_HighAccuracy):
    offline_expected_qps = 5000 * 8


class H100_PCIe_80GBx8_Triton(H100_PCIe_80GBx8):
    use_triton = True


class H100_PCIe_80GBx8_HighAccuracy_Triton(H100_PCIe_80GBx8_HighAccuracy):
    use_triton = True


class H100_PCIe_80GB_aarch64x1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = False
    gpu_batch_size = {'bert': 1280}
    offline_expected_qps = 4000
    workspace_size = 7516192768


class H100_PCIe_80GB_aarch64x1_HighAccuracy(H100_PCIe_80GB_aarch64x1):
    precision = "fp16"
    offline_expected_qps = 1800
    use_graphs = False
    gpu_batch_size = {'bert': 1024}


class H100_PCIe_80GB_aarch64x4(H100_PCIe_80GB_aarch64x1):
    offline_expected_qps = 16000


class H100_PCIe_80GB_aarch64x4_HighAccuracy(H100_PCIe_80GB_aarch64x1_HighAccuracy):
    offline_expected_qps = 8000


class H100_SXM_80GBx1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = False
    gpu_batch_size = {'bert': 1280}
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    offline_expected_qps = 9400
    workspace_size = 7516192768


class H100_SXM_80GBx1_HighAccuracy(H100_SXM_80GBx1):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = {'bert': 1024}
    offline_expected_qps = 8200


class H100_SXM_80GBx1_Triton(H100_SXM_80GBx1):
    use_triton = True


class H100_SXM_80GBx1_HighAccuracy_Triton(H100_SXM_80GBx1_HighAccuracy):
    use_triton = True


class H100_SXM_80GBx8(H100_SXM_80GBx1):
    offline_expected_qps = 9400 * 8


class H100_SXM_80GBx8_MaxQ(H100_SXM_80GBx8):
    offline_expected_qps = 54000
    power_limit = 400


class H100_SXM_80GBx8_HighAccuracy(H100_SXM_80GBx1_HighAccuracy):
    offline_expected_qps = 8200 * 8


class H100_SXM_80GBx8_HighAccuracy_MaxQ(H100_SXM_80GBx8_HighAccuracy):
    power_limit = 450
    offline_expected_qps = 51000


class H100_SXM_80GBx8_Triton(H100_SXM_80GBx8):
    use_triton = True


class H100_SXM_80GBx8_HighAccuracy_Triton(H100_SXM_80GBx8_HighAccuracy):
    use_triton = True


class H200_SXM_141GBx1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = False
    gpu_batch_size = {'bert': 1280}
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    offline_expected_qps = 9400
    workspace_size = 128000000000


class H200_SXM_141GBx8(H200_SXM_141GBx1):
    offline_expected_qps = 9400 * 8


class H200_SXM_141GBx1_HighAccuracy(H200_SXM_141GBx1):
    gpu_batch_size = {'bert': 1024}
    precision = "fp16"
    use_fp8 = True
    offline_expected_qps = 8200


class H200_SXM_141GBx8_HighAccuracy(H200_SXM_141GBx1_HighAccuracy):
    offline_expected_qps = 8200 * 8


class H200_SXM_141GB_CTSx1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = False
    gpu_batch_size = {'bert': 1280}
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    offline_expected_qps = 9400
    workspace_size = 128000000000


class H200_SXM_141GB_CTSx8(H200_SXM_141GB_CTSx1):
    offline_expected_qps = 9400 * 8


class H200_SXM_141GB_CTSx1_HighAccuracy(H200_SXM_141GB_CTSx1):
    gpu_batch_size = {'bert': 1024}
    precision = "fp16"
    use_fp8 = True
    offline_expected_qps = 8200


class H200_SXM_141GB_CTSx8_HighAccuracy(H200_SXM_141GB_CTSx1_HighAccuracy):
    offline_expected_qps = 8200 * 8


class L4x1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = {'bert': 16}
    energy_aware_kernels = True
    offline_expected_qps = 1000
    workspace_size = 7516192768


class L4x1_HighAccuracy(L4x1):
    precision = "fp16"
    use_fp8 = True
    gpu_batch_size = {'bert': 16}
    offline_expected_qps = 640
    gpu_inference_streams = 1
    energy_aware_kernels = False
    gpu_copy_streams = 1


class L40x1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = {'bert': 1024}
    offline_expected_qps = 3400
    workspace_size = 7516192768


class L40Sx1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = {'bert': 32}
    offline_expected_qps = 3400
    workspace_size = 7516192768


class L40x1_HighAccuracy(L40x1):
    precision = "fp16"
    offline_expected_qps = 1750


class L40Sx1_HighAccuracy(OfflineGPUBaseConfig):
    precision = "fp16"
    use_fp8 = True
    offline_expected_qps = 3300
    gpu_batch_size = {'bert': 32}


class A100_PCIe_80GBx1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = {'bert': 1024}
    offline_expected_qps = 3400
    workspace_size = 7516192768


class A100_PCIe_80GBx1_HighAccuracy(A100_PCIe_80GBx1):
    precision = "fp16"
    offline_expected_qps = 1750


class A100_PCIe_80GBx1_Triton(A100_PCIe_80GBx1):
    use_triton = True
    offline_expected_qps = 3000


class A100_PCIe_80GBx1_TritonUnified(A100_PCIe_80GBx1):
    use_triton = True
    offline_expected_qps = 3000


class A100_PCIe_80GBx1_HighAccuracy_Triton(A100_PCIe_80GBx1_HighAccuracy):
    use_triton = True
    offline_expected_qps = 1550


class A100_PCIe_80GBx1_HighAccuracy_TritonUnified(A100_PCIe_80GBx1_HighAccuracy):
    use_triton = True
    offline_expected_qps = 1550


class A100_PCIe_80GBx8(A100_PCIe_80GBx1):
    offline_expected_qps = 27200


class A100_PCIe_80GBx8_HighAccuracy(A100_PCIe_80GBx8):
    precision = "fp16"
    offline_expected_qps = 12800


class A100_PCIe_80GBx8_Triton(A100_PCIe_80GBx8):
    use_triton = True
    offline_expected_qps = 27000


class A100_PCIe_80GBx8_TritonUnified(A100_PCIe_80GBx8):
    use_triton = True
    offline_expected_qps = 27000


class A100_PCIe_80GBx8_HighAccuracy_Triton(A100_PCIe_80GBx8_HighAccuracy):
    use_triton = True
    offline_expected_qps = 12800


class A100_PCIe_80GBx8_HighAccuracy_TritonUnified(A100_PCIe_80GBx8_HighAccuracy):
    use_triton = True
    offline_expected_qps = 12800


class A100_PCIe_80GBx8_MaxQ(A100_PCIe_80GBx8):
    offline_expected_qps = 27200
    power_limit = 240


class A100_PCIe_80GBx8_HighAccuracy_MaxQ(A100_PCIe_80GBx8_MaxQ):
    precision = "fp16"
    offline_expected_qps = 11000


class A100_PCIe_80GBx8_Triton_MaxQ(A100_PCIe_80GBx8_MaxQ):
    use_triton = True
    offline_expected_qps = 27200


class A100_PCIe_80GBx8_TritonUnified_MaxQ(A100_PCIe_80GBx8_MaxQ):
    use_triton = True
    offline_expected_qps = 27200


class A100_PCIe_80GBx8_HighAccuracy_Triton_MaxQ(A100_PCIe_80GBx8_HighAccuracy_MaxQ):
    use_triton = True
    offline_expected_qps = 11168


class A100_PCIe_80GBx8_HighAccuracy_TritonUnified_MaxQ(A100_PCIe_80GBx8_HighAccuracy_MaxQ):
    use_triton = True
    offline_expected_qps = 11168


class A100_PCIe_80GB_aarch64x1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = {'bert': 1024}
    offline_expected_qps = 3400
    workspace_size = 7516192768


class A100_PCIe_80GB_aarch64x1_Triton(A100_PCIe_80GB_aarch64x1):
    use_triton = True


class A100_PCIe_80GB_aarch64x1_TritonUnified(A100_PCIe_80GB_aarch64x1):
    use_triton = True


class A100_PCIe_80GB_aarch64x1_HighAccuracy(A100_PCIe_80GB_aarch64x1):
    precision = "fp16"
    offline_expected_qps = 1950


class A100_PCIe_80GB_aarch64x1_HighAccuracy_Triton(A100_PCIe_80GB_aarch64x1_HighAccuracy):
    use_triton = True


class A100_PCIe_80GB_aarch64x1_HighAccuracy_TritonUnified(A100_PCIe_80GB_aarch64x1_HighAccuracy):
    use_triton = True


class A100_PCIe_80GB_aarch64x2(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = {'bert': 1024}
    offline_expected_qps = 6500
    workspace_size = 7516192768


class A100_PCIe_80GB_aarch64x2_Triton(A100_PCIe_80GB_aarch64x2):
    use_triton = True


class A100_PCIe_80GB_aarch64x2_TritonUnified(A100_PCIe_80GB_aarch64x2):
    use_triton = True


class A100_PCIe_80GB_aarch64x2_HighAccuracy(A100_PCIe_80GB_aarch64x2):
    precision = "fp16"
    offline_expected_qps = 3900


class A100_PCIe_80GB_aarch64x2_HighAccuracy_Triton(A100_PCIe_80GB_aarch64x2_HighAccuracy):
    use_triton = True


class A100_PCIe_80GB_aarch64x2_HighAccuracy_TritonUnified(A100_PCIe_80GB_aarch64x2_HighAccuracy):
    use_triton = True


class A100_PCIe_80GB_aarch64x4(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = {'bert': 1024}
    offline_expected_qps = 13600
    workspace_size = 7516192768


class A100_PCIe_80GB_aarch64x4_Triton(A100_PCIe_80GB_aarch64x4):
    use_triton = True


class A100_PCIe_80GB_aarch64x4_TritonUnified(A100_PCIe_80GB_aarch64x4):
    use_triton = True


class A100_PCIe_80GB_aarch64x4_HighAccuracy(A100_PCIe_80GB_aarch64x4):
    precision = "fp16"
    offline_expected_qps = 8200


class A100_PCIe_80GB_aarch64x4_HighAccuracy_Triton(A100_PCIe_80GB_aarch64x4_HighAccuracy):
    use_triton = True


class A100_PCIe_80GB_aarch64x4_HighAccuracy_TritonUnified(A100_PCIe_80GB_aarch64x4_HighAccuracy):
    use_triton = True


class A100_PCIe_80GB_aarch64x4_MaxQ(A100_PCIe_80GB_aarch64x4):
    offline_expected_qps = 10000
    power_limit = 225


class A100_PCIe_80GB_aarch64x4_HighAccuracy_MaxQ(A100_PCIe_80GB_aarch64x4_MaxQ):
    precision = "fp16"
    offline_expected_qps = 5000


class A100_PCIe_aarch64x1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = {'bert': 1024}
    offline_expected_qps = 3400
    workspace_size = 7516192768


class A100_PCIe_aarch64x1_Triton(A100_PCIe_aarch64x1):
    use_triton = True


class A100_PCIe_aarch64x1_TritonUnified(A100_PCIe_aarch64x1):
    use_triton = True


class A100_PCIe_aarch64x1_HighAccuracy(A100_PCIe_aarch64x1):
    precision = "fp16"
    offline_expected_qps = 1950


class A100_PCIe_aarch64x1_HighAccuracy_Triton(A100_PCIe_aarch64x1_HighAccuracy):
    use_triton = True


class A100_PCIe_aarch64x1_HighAccuracy_TritonUnified(A100_PCIe_aarch64x1_HighAccuracy):
    use_triton = True


class A100_PCIe_aarch64x2(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = {'bert': 1024}
    offline_expected_qps = 6500
    workspace_size = 7516192768


class A100_PCIe_aarch64x2_Triton(A100_PCIe_aarch64x2):
    use_triton = True


class A100_PCIe_aarch64x2_TritonUnified(A100_PCIe_aarch64x2):
    use_triton = True


class A100_PCIe_aarch64x2_HighAccuracy(A100_PCIe_aarch64x2):
    precision = "fp16"
    offline_expected_qps = 3900


class A100_PCIe_aarch64x2_HighAccuracy_Triton(A100_PCIe_aarch64x2_HighAccuracy):
    use_triton = True


class A100_PCIe_aarch64x2_HighAccuracy_TritonUnified(A100_PCIe_aarch64x2_HighAccuracy):
    use_triton = True


class A100_PCIe_aarch64x4(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = {'bert': 1024}
    offline_expected_qps = 13600
    workspace_size = 7516192768


class A100_PCIe_aarch64x4_Triton(A100_PCIe_aarch64x4):
    use_triton = True


class A100_PCIe_aarch64x4_TritonUnified(A100_PCIe_aarch64x4):
    use_triton = True


class A100_PCIe_aarch64x4_HighAccuracy(A100_PCIe_aarch64x4):
    precision = "fp16"
    offline_expected_qps = 8200


class A100_PCIe_aarch64x4_HighAccuracy_Triton(A100_PCIe_aarch64x4_HighAccuracy):
    use_triton = True


class A100_PCIe_aarch64x4_HighAccuracy_TritonUnified(A100_PCIe_aarch64x4_HighAccuracy):
    use_triton = True


class A100_PCIe_aarch64x4_MaxQ(A100_PCIe_aarch64x4):
    offline_expected_qps = 9000
    power_limit = 225


class A100_PCIe_aarch64x4_HighAccuracy_MaxQ(A100_PCIe_aarch64x4_MaxQ):
    precision = "fp16"
    offline_expected_qps = 4500


class A100_PCIe_MIG_1x1g5gb(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = {'bert': 64}
    offline_expected_qps = 500
    workspace_size = 2147483648


class A100_PCIe_MIG_1x1g5gb_HighAccuracy(A100_PCIe_MIG_1x1g5gb):
    precision = "fp16"
    offline_expected_qps = 225


class A100_PCIe_MIG_1x1g5gb_Triton(A100_PCIe_MIG_1x1g5gb):
    use_triton = True


class A100_PCIe_MIG_1x1g5gb_TritonUnified(A100_PCIe_MIG_1x1g5gb):
    use_triton = True


class A100_PCIe_MIG_1x1g5gb_HighAccuracy_Triton(A100_PCIe_MIG_1x1g5gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = {'bert': 32}
    offline_expected_qps = 225


class A100_PCIe_MIG_1x1g5gb_HighAccuracy_TritonUnified(A100_PCIe_MIG_1x1g5gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = {'bert': 32}
    offline_expected_qps = 225


class A100_PCIe_80GB_MIG_1x1g10gb(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = {'bert': 64}
    offline_expected_qps = 500
    workspace_size = 2147483648


class A100_PCIe_80GB_MIG_1x1g10gb_Hetero(A100_PCIe_80GB_MIG_1x1g10gb):
    offline_expected_qps = 470


class A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy(A100_PCIe_80GB_MIG_1x1g10gb):
    precision = "fp16"
    offline_expected_qps = 225


class A100_PCIe_80GB_MIG_1x1g10gb_Hetero_HighAccuracy(A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy):
    offline_expected_qps = 210


class A100_PCIe_80GB_MIG_1x1g10gb_Triton(A100_PCIe_80GB_MIG_1x1g10gb):
    use_triton = True


class A100_PCIe_80GB_MIG_1x1g10gb_TritonUnified(A100_PCIe_80GB_MIG_1x1g10gb):
    use_triton = True


class A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy_Triton(A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = {'bert': 32}
    offline_expected_qps = 225


class A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy_TritonUnified(A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = {'bert': 32}
    offline_expected_qps = 225


class A100_SXM_80GB_MIG_1x1g10gb(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = {'bert': 64}
    offline_expected_qps = 500
    workspace_size = 2147483648


class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    pass


class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy(A100_SXM_80GB_MIG_1x1g10gb):
    precision = "fp16"
    offline_expected_qps = 225


class A100_SXM_80GB_MIG_1x1g10gb_Hetero_HighAccuracy(A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy):
    pass


class A100_SXM_80GB_MIG_1x1g10gb_Triton(A100_SXM_80GB_MIG_1x1g10gb):
    use_triton = True


class A100_SXM_80GB_MIG_1x1g10gb_TritonUnified(A100_SXM_80GB_MIG_1x1g10gb):
    use_triton = True


class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy_Triton(A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = {'bert': 32}
    offline_expected_qps = 225


class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy_TritonUnified(A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = {'bert': 32}
    offline_expected_qps = 225


class A100_SXM_80GBx1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = {'bert': 1280}
    gpu_inference_streams = 1
    offline_expected_qps = 3500


class A100_SXM_80GBx1_HighAccuracy(A100_SXM_80GBx1):
    precision = "fp16"
    gpu_batch_size = {'bert': 512}
    offline_expected_qps = 1750


class A100_SXM_80GBx1_Triton(A100_SXM_80GBx1):
    use_triton = True


class A100_SXM_80GBx1_TritonUnified(A100_SXM_80GBx1):
    use_triton = True


class A100_SXM_80GBx1_HighAccuracy_Triton(A100_SXM_80GBx1_HighAccuracy):
    use_triton = True
    gpu_batch_size = {'bert': 1280}
    offline_expected_qps = 1750


class A100_SXM_80GBx1_HighAccuracy_TritonUnified(A100_SXM_80GBx1_HighAccuracy):
    use_triton = True
    gpu_batch_size = {'bert': 1280}
    offline_expected_qps = 1750


class A100_SXM_80GBx8(A100_SXM_80GBx1):
    offline_expected_qps = 30000
    workspace_size = 7516192768


class A100_SXM_80GBx8_HighAccuracy(A100_SXM_80GBx8):
    precision = "fp16"
    gpu_batch_size = {'bert': 512}
    offline_expected_qps = 15000


class A100_SXM_80GBx8_Triton(A100_SXM_80GBx8):
    use_triton = True
    offline_expected_qps = 29000
    workspace_size = 7516192768
    batch_triton_requests = False


class A100_SXM_80GBx8_TritonUnified(A100_SXM_80GBx8):
    use_triton = True
    offline_expected_qps = 29000
    workspace_size = 7516192768
    batch_triton_requests = False


class A100_SXM_80GBx8_HighAccuracy_Triton(A100_SXM_80GBx8_HighAccuracy):
    use_triton = True
    gpu_batch_size = {'bert': 1280}
    gpu_inference_streams = 1
    offline_expected_qps = 15000


class A100_SXM_80GBx8_HighAccuracy_TritonUnified(A100_SXM_80GBx8_HighAccuracy):
    use_triton = True
    gpu_batch_size = {'bert': 1280}
    gpu_inference_streams = 1
    offline_expected_qps = 15000


class A100_SXM_80GBx8_MaxQ(A100_SXM_80GBx8):
    power_limit = 275


class A100_SXM_80GBx8_HighAccuracy_MaxQ(A100_SXM_80GBx8_MaxQ):
    power_limit = 275
    precision = "fp16"
    gpu_batch_size = {'bert': 512}
    offline_expected_qps = 11000


class A100_SXM_80GBx8_Triton_MaxQ(A100_SXM_80GBx8_MaxQ):
    use_triton = True


class A100_SXM_80GBx8_TritonUnified_MaxQ(A100_SXM_80GBx8_MaxQ):
    use_triton = True


class A100_SXM_80GBx8_HighAccuracy_Triton_MaxQ(A100_SXM_80GBx8_HighAccuracy_MaxQ):
    use_triton = True


class A100_SXM_80GBx8_HighAccuracy_TritonUnified_MaxQ(A100_SXM_80GBx8_HighAccuracy_MaxQ):
    use_triton = True


class A100_SXM_80GB_aarch64x1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = {'bert': 1280}
    gpu_inference_streams = 1
    offline_expected_qps = 2500


class A100_SXM_80GB_aarch64x1_HighAccuracy(A100_SXM_80GB_aarch64x1):
    precision = "fp16"
    gpu_batch_size = {'bert': 512}
    offline_expected_qps = 1750


class A100_SXM_80GB_aarch64x1_Triton(A100_SXM_80GB_aarch64x1):
    use_triton = True
    offline_expected_qps = 2200


class A100_SXM_80GB_aarch64x1_TritonUnified(A100_SXM_80GB_aarch64x1):
    use_triton = True
    offline_expected_qps = 2200


class A100_SXM_80GB_aarch64x1_HighAccuracy_Triton(A100_SXM_80GB_aarch64x1_HighAccuracy):
    use_triton = True
    gpu_batch_size = {'bert': 1280}
    offline_expected_qps = 1750


class A100_SXM_80GB_aarch64x1_HighAccuracy_TritonUnified(A100_SXM_80GB_aarch64x1_HighAccuracy):
    use_triton = True
    gpu_batch_size = {'bert': 1280}
    offline_expected_qps = 1750


class A100_SXM_80GB_aarch64x8(A100_SXM_80GB_aarch64x1):
    offline_expected_qps = 27500
    workspace_size = 7516192768


class A100_SXM_80GB_aarch64x8_MaxQ(A100_SXM_80GB_aarch64x8):
    offline_expected_qps = 22000
    power_limit = 250           # Set to 250 initially, increase to 300 w/ optimal fan setting


class A100_SXM_80GB_aarch64x8_HighAccuracy(A100_SXM_80GB_aarch64x8):
    precision = "fp16"
    gpu_batch_size = {'bert': 512}
    offline_expected_qps = 14000


class A100_SXM_80GB_aarch64x8_HighAccuracy_MaxQ(A100_SXM_80GB_aarch64x8_HighAccuracy):
    offline_expected_qps = 12000
    power_limit = 250           # Set to 250 initially, increase to 300 w/ optimal fan setting


class A100_SXM_80GB_aarch64x8_Triton(A100_SXM_80GB_aarch64x8):
    use_triton = True
    offline_expected_qps = 27500
    workspace_size = 7516192768
    batch_triton_requests = False


class A100_SXM_80GB_aarch64x8_TritonUnified(A100_SXM_80GB_aarch64x8):
    use_triton = True
    offline_expected_qps = 27500
    workspace_size = 7516192768
    batch_triton_requests = False


class A100_SXM_80GB_aarch64x8_HighAccuracy_Triton(A100_SXM_80GB_aarch64x8_HighAccuracy):
    use_triton = True
    gpu_batch_size = {'bert': 1280}
    gpu_inference_streams = 1
    offline_expected_qps = 14000


class A100_SXM_80GB_aarch64x8_HighAccuracy_TritonUnified(A100_SXM_80GB_aarch64x8_HighAccuracy):
    use_triton = True
    gpu_batch_size = {'bert': 1280}
    gpu_inference_streams = 1
    offline_expected_qps = 14000


class A100_SXM_80GB_aarch64_MIG_1x1g10gb(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = {'bert': 64}
    offline_expected_qps = 350
    workspace_size = 2147483648


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Hetero(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    pass


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    precision = "fp16"
    offline_expected_qps = 250


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Hetero_HighAccuracy(A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy):
    pass


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Triton(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    use_triton = True


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_TritonUnified(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    use_triton = True


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy_Triton(A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = {'bert': 32}
    offline_expected_qps = 250


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy_TritonUnified(A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = {'bert': 32}
    offline_expected_qps = 250


class A100_SXM4_40GB_MIG_1x1g5gb(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = {'bert': 64}
    offline_expected_qps = 500
    workspace_size = 2147483648


class A100_SXM4_40GB_MIG_1x1g5gb_HighAccuracy(A100_SXM4_40GB_MIG_1x1g5gb):
    precision = "fp16"
    gpu_batch_size = {'bert': 64}
    offline_expected_qps = 225


class A100_SXM4_40GB_MIG_1x1g5gb_Triton(A100_SXM4_40GB_MIG_1x1g5gb):
    use_triton = True


class A100_SXM4_40GB_MIG_1x1g5gb_TritonUnified(A100_SXM4_40GB_MIG_1x1g5gb):
    use_triton = True


class A100_SXM4_40GB_MIG_1x1g5gb_HighAccuracy_Triton(A100_SXM4_40GB_MIG_1x1g5gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = {'bert': 32}
    offline_expected_qps = 225


class A100_SXM4_40GB_MIG_1x1g5gb_HighAccuracy_TritonUnified(A100_SXM4_40GB_MIG_1x1g5gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = {'bert': 32}
    offline_expected_qps = 225


class A100_SXM4_40GBx1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = {'bert': 1024}
    offline_expected_qps = 3400


class A100_SXM4_40GBx1_HighAccuracy(A100_SXM4_40GBx1):
    precision = "fp16"
    offline_expected_qps = 1750


class A100_SXM4_40GBx1_Triton(A100_SXM4_40GBx1):
    use_triton = True


class A100_SXM4_40GBx1_TritonUnified(A100_SXM4_40GBx1):
    use_triton = True


class A100_SXM4_40GBx1_HighAccuracy_Triton(A100_SXM4_40GBx1_HighAccuracy):
    use_triton = True


class A100_SXM4_40GBx1_HighAccuracy_TritonUnified(A100_SXM4_40GBx1_HighAccuracy):
    use_triton = True


class A100_SXM4_40GBx8(A100_SXM4_40GBx1):
    offline_expected_qps = 30000
    workspace_size = 7516192768


class A100_SXM4_40GBx8_HighAccuracy(A100_SXM4_40GBx8):
    precision = "fp16"
    gpu_batch_size = {'bert': 1024}
    offline_expected_qps = 15000


class A100_SXM4_40GBx8_Triton(A100_SXM4_40GBx8):
    use_triton = True


class A100_SXM4_40GBx8_TritonUnified(A100_SXM4_40GBx8):
    use_triton = True


class A100_SXM4_40GBx8_HighAccuracy_Triton(A100_SXM4_40GBx8_HighAccuracy):
    use_triton = True


class A100_SXM4_40GBx8_HighAccuracy_TritonUnified(A100_SXM4_40GBx8_HighAccuracy):
    use_triton = True


class Orin(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = {'bert': 256}
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    offline_expected_qps = 550


class Orin_Triton(Orin):
    use_triton = True
    batch_triton_requests = True


class Orin_TritonUnified(Orin):
    use_triton = True
    batch_triton_requests = True


class Orin_MaxQ(Orin):
    # NOTE: Orin AGX 3.1 Shmoo
    use_small_tile_gemm_plugin = True
    gpu_batch_size = {'bert': 384}
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    soc_cpu_freq = 576000
    soc_gpu_freq = 714000000
    soc_dla_freq = 0
    soc_emc_freq = 2133000000
    soc_pva_freq = 115000000
    orin_num_cores = 4
    offline_expected_qps = 300


class Orin_NX(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = {'bert': 256}
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    offline_expected_qps = 190


class Orin_NX_MaxQ(Orin_NX):
    # NOTE: Orin NX 3.1 Shmoo
    use_small_tile_gemm_plugin = True
    gpu_batch_size = {'bert': 384}
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    soc_cpu_freq = 499200
    soc_gpu_freq = 714000000
    soc_dla_freq = 0
    soc_emc_freq = 2133000000
    soc_pva_freq = 0
    orin_num_cores = 4
    offline_expected_qps = 140


class H100_PCIe_80GBx8_MaxQ(H100_PCIe_80GBx8):
    offline_expected_qps = 39500
    power_limit = 290


class H100_PCIe_80GBx8_HighAccuracy_MaxQ(H100_PCIe_80GBx8_HighAccuracy):
    offline_expected_qps = 33000
    power_limit = 300
