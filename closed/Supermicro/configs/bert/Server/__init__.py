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


class GH200_96GB_aarch64x1(ServerGPUBaseConfig):
    use_small_tile_gemm_plugin = False
    use_graphs = False
    gpu_batch_size = {'bert': 256}
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 7700
    server_num_issue_query_threads = 1
    workspace_size = 7516192768


class GH200_96GB_aarch64x1_High_Accuracy(GH200_96GB_aarch64x1):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = {'bert': 512}
    server_target_qps = 7000


class H200_SXM_141GBx1(ServerGPUBaseConfig):
    use_small_tile_gemm_plugin = False
    use_graphs = False
    gpu_batch_size = {'bert': 128}
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    server_target_qps = 7100
    server_num_issue_query_threads = 1
    workspace_size = 7516192768


class H200_SXM_141GBx8(H200_SXM_141GBx1):
    gpu_inference_streams = 2
    gpu_copy_streams = 4
    server_target_qps = H200_SXM_141GBx1.server_target_qps * 8


class H200_SXM_141GBx8_MaxQ(H200_SXM_141GBx8):
    power_limit = 400
    server_target_qps = 5300 * 8


class H200_SXM_141GBx1_HighAccuracy(H200_SXM_141GBx1):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = {'bert': 512}
    server_target_qps = 6100


class H200_SXM_141GBx8_HighAccuracy(H200_SXM_141GBx8):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = {'bert': 512}
    gpu_inference_streams = 1
    server_target_qps = 6400 * 8


class H200_SXM_141GBx8_HighAccuracy_MaxQ(H200_SXM_141GBx8_HighAccuracy):
    power_limit = 450
    server_target_qps = 4900 * 8


class H100_PCIe_80GBx1(ServerGPUBaseConfig):
    use_small_tile_gemm_plugin = False
    use_graphs = False
    graphs_max_seqlen = 200
    gpu_batch_size = {'bert': 256}
    server_target_qps = 4560
    server_num_issue_query_threads = 1
    workspace_size = 7516192768
    # soft_drop = 1.0
    soft_drop = 0.99


class H100_PCIe_80GBx1_HighAccuracy(H100_PCIe_80GBx1):
    precision = "fp16"
    use_fp8 = True
    server_target_qps = 4000
    soft_drop = 1.0


class H100_PCIe_80GBx1_Triton(H100_PCIe_80GBx1):
    use_triton = True
    server_target_qps = 2300


class H100_PCIe_80GBx1_HighAccuracy_Triton(H100_PCIe_80GBx1_HighAccuracy):
    use_triton = True
    server_target_qps = 1150


class H100_PCIe_80GBx8(ServerGPUBaseConfig):
    use_small_tile_gemm_plugin = False
    use_graphs = False
    gpu_batch_size = {'bert': 64}
    server_target_qps = 4420 * 8
    server_num_issue_query_threads = 1
    workspace_size = 7516192768
    graphs_max_seqlen = 200
    soft_drop = 1.0


class H100_PCIe_80GBx8_MaxQ(H100_PCIe_80GBx8):
    power_limit = 310
    server_target_qps = 33000


class H100_PCIe_80GBx8_HighAccuracy(H100_PCIe_80GBx1_HighAccuracy):
    gpu_batch_size = {'bert': 512}
    server_target_qps = 4000 * 8


class H100_PCIe_80GBx8_HighAccuracy_MaxQ(H100_PCIe_80GBx8_HighAccuracy):
    power_limit = 300
    server_target_qps = 28500


class H100_PCIe_80GBx8_Triton(H100_PCIe_80GBx8):
    use_triton = True
    server_target_qps = 18500


class H100_PCIe_80GBx8_HighAccuracy_Triton(H100_PCIe_80GBx8_HighAccuracy):
    use_triton = True
    server_target_qps = 9250


class H100_PCIe_80GB_aarch64x1(ServerGPUBaseConfig):
    use_small_tile_gemm_plugin = False
    use_graphs = False
    gpu_batch_size = {'bert': 1280}
    server_target_qps = 3000
    server_num_issue_query_threads = 1
    workspace_size = 7516192768


class H100_PCIe_80GB_aarch64x1_HighAccuracy(H100_PCIe_80GB_aarch64x1):
    precision = "fp16"
    server_target_qps = 1500


class H100_PCIe_80GB_aarch64x4(ServerGPUBaseConfig):
    use_small_tile_gemm_plugin = False
    use_graphs = False
    gpu_batch_size = {'bert': 1280}
    server_target_qps = 12000
    server_num_issue_query_threads = 1
    workspace_size = 7516192768


class H100_PCIe_80GB_aarch64x4_HighAccuracy(H100_PCIe_80GB_aarch64x4):
    precision = "fp16"
    server_target_qps = 6000


class H100_SXM_80GBx1(ServerGPUBaseConfig):
    use_small_tile_gemm_plugin = False
    use_graphs = False
    gpu_batch_size = {'bert': 128}
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    server_target_qps = 7000
    server_num_issue_query_threads = 1
    workspace_size = 7516192768


class H100_SXM_80GBx8(H100_SXM_80GBx1):
    gpu_inference_streams = 2
    gpu_copy_streams = 4
    server_target_qps = H100_SXM_80GBx1.server_target_qps * 8


class H100_SXM_80GBx8_MaxQ(H100_SXM_80GBx8):
    power_limit = 400
    server_target_qps = 5300 * 8


class H100_SXM_80GBx1_HighAccuracy(H100_SXM_80GBx1):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = {'bert': 512}
    server_target_qps = 6100


class H100_SXM_80GBx8_HighAccuracy(H100_SXM_80GBx8):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = {'bert': 512}
    gpu_inference_streams = 1
    server_target_qps = 6200 * 8


class H100_SXM_80GBx8_HighAccuracy_MaxQ(H100_SXM_80GBx8_HighAccuracy):
    power_limit = 450
    server_target_qps = 4900 * 8


class L4x1(ServerGPUBaseConfig):
    gpu_batch_size = {'bert': 16}
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 900
    soft_drop = 1.0
    energy_aware_kernels = True
    use_small_tile_gemm_plugin = True


class L4x1_HighAccuracy(L4x1):
    precision = "fp16"
    gpu_batch_size = {'bert': 16}
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 540
    soft_drop = 1.0
    use_fp8 = True
    use_graphs = False
    energy_aware_kernels = False
    use_small_tile_gemm_plugin = False


class L40x1(ServerGPUBaseConfig):
    gpu_batch_size = {'bert': 64}
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 1500.0
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True


class L40x1_HighAccuracy(L40x1):
    precision = "fp16"
    server_target_qps = 1500


class A100_PCIe_80GBx8(ServerGPUBaseConfig):
    active_sms = 60
    gpu_batch_size = {'bert': 64}
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 23000.0
    soft_drop = 1.0


class A100_PCIe_80GBx8_HighAccuracy(A100_PCIe_80GBx8):
    precision = "fp16"
    server_target_qps = 10800


class A100_PCIe_80GBx8_Triton(A100_PCIe_80GBx8):
    server_target_qps = 18000
    use_triton = True


class A100_PCIe_80GBx8_TritonUnified(A100_PCIe_80GBx8):
    server_target_qps = 18000
    use_triton = True


class A100_PCIe_80GBx8_HighAccuracy_Triton(A100_PCIe_80GBx8_HighAccuracy):
    server_target_qps = 9500
    use_triton = True


class A100_PCIe_80GBx8_HighAccuracy_TritonUnified(A100_PCIe_80GBx8_HighAccuracy):
    server_target_qps = 9500
    use_triton = True


class A100_PCIe_80GBx8_MaxQ(A100_PCIe_80GBx8):
    server_target_qps = 17300
    power_limit = 200


class A100_PCIe_80GBx8_HighAccuracy_MaxQ(A100_PCIe_80GBx8_MaxQ):
    precision = "fp16"
    server_target_qps = 7500
    power_limit = 200


class A100_PCIe_80GBx8_Triton_MaxQ(A100_PCIe_80GBx8_MaxQ):
    server_target_qps = 17000
    use_triton = True


class A100_PCIe_80GBx8_TritonUnified_MaxQ(A100_PCIe_80GBx8_MaxQ):
    server_target_qps = 17000
    use_triton = True


class A100_PCIe_80GBx8_HighAccuracy_Triton_MaxQ(A100_PCIe_80GBx8_HighAccuracy_MaxQ):
    server_target_qps = 9480
    use_triton = True


class A100_PCIe_80GB_aarch64x4(ServerGPUBaseConfig):
    active_sms = 60
    gpu_batch_size = {'bert': 64}
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 10400.0
    soft_drop = 1.0


class A100_PCIe_80GB_aarch64x4_Triton(A100_PCIe_80GB_aarch64x4):
    use_triton = True
    server_target_qps = 10000


class A100_PCIe_80GB_aarch64x4_TritonUnified(A100_PCIe_80GB_aarch64x4):
    use_triton = True
    server_target_qps = 10000


class A100_PCIe_80GB_aarch64x4_HighAccuracy(A100_PCIe_80GB_aarch64x4):
    precision = "fp16"
    server_target_qps = 4800


class A100_PCIe_80GB_aarch64x4_HighAccuracy_Triton(A100_PCIe_80GB_aarch64x4_HighAccuracy):
    use_triton = True
    server_target_qps = 4500


class A100_PCIe_80GB_aarch64x4_HighAccuracy_TritonUnified(A100_PCIe_80GB_aarch64x4_HighAccuracy):
    use_triton = True
    server_target_qps = 4500


class A100_PCIe_80GB_aarch64x4_MaxQ(A100_PCIe_80GB_aarch64x4):
    server_target_qps = 10000.0
    power_limit = 225


class A100_PCIe_80GB_aarch64x4_HighAccuracy_MaxQ(A100_PCIe_80GB_aarch64x4_MaxQ):
    precision = "fp16"
    server_target_qps = 4000


class A100_PCIe_80GB_MIG_1x1g10gb(ServerGPUBaseConfig):
    gpu_inference_streams = 1
    active_sms = 100
    gpu_batch_size = {'bert': 16}
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 0
    server_target_qps = 380
    soft_drop = 0.99
    use_graphs = False


class A100_PCIe_80GB_MIG_1x1g10gb_Hetero(A100_PCIe_80GB_MIG_1x1g10gb):
    server_target_qps = 360


class A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy(A100_PCIe_80GB_MIG_1x1g10gb):
    precision = "fp16"
    server_target_qps = 170


class A100_PCIe_80GB_MIG_1x1g10gb_Hetero_HighAccuracy(A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy):
    server_target_qps = 160


class A100_PCIe_80GB_MIG_1x1g10gb_Triton(A100_PCIe_80GB_MIG_1x1g10gb):
    server_target_qps = 360
    use_triton = True


class A100_PCIe_80GB_MIG_1x1g10gb_TritonUnified(A100_PCIe_80GB_MIG_1x1g10gb):
    server_target_qps = 360
    use_triton = True


class A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy_Triton(A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy):
    precision = "fp16"
    server_target_qps = 170
    use_triton = True


class A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy_TritonUnified(A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy):
    precision = "fp16"
    server_target_qps = 170
    use_triton = True


class A100_SXM_80GB_MIG_1x1g10gb(ServerGPUBaseConfig):
    gpu_inference_streams = 1
    active_sms = 100
    gpu_batch_size = {'bert': 16}
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 0
    server_target_qps = 380
    soft_drop = 0.993
    deque_timeout_usec = 50000
    use_graphs = False


class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    pass


class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy(A100_SXM_80GB_MIG_1x1g10gb):
    precision = "fp16"
    server_target_qps = 164


class A100_SXM_80GB_MIG_1x1g10gb_Hetero_HighAccuracy(A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy):
    server_target_qps = 160


class A100_SXM_80GB_MIG_1x1g10gb_Triton(A100_SXM_80GB_MIG_1x1g10gb):
    server_target_qps = 360
    use_triton = True


class A100_SXM_80GB_MIG_1x1g10gb_TritonUnified(A100_SXM_80GB_MIG_1x1g10gb):
    server_target_qps = 360
    use_triton = True


class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy_Triton(A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy):
    precision = "fp16"
    gpu_batch_size = {'bert': 8}
    server_target_qps = 166
    use_triton = True


class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy_TritonUnified(A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy):
    precision = "fp16"
    gpu_batch_size = {'bert': 8}
    server_target_qps = 170
    use_triton = True


class A100_SXM_80GBx8(ServerGPUBaseConfig):
    active_sms = 60
    gpu_batch_size = {'bert': 48}
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 25400
    soft_drop = 1.0
    gpu_copy_streams = 4
    gpu_inference_streams = 2


class A100_SXM_80GBx8_HighAccuracy(A100_SXM_80GBx8):
    gpu_batch_size = {'bert': 24}
    precision = "fp16"
    server_target_qps = 12820
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    soft_drop = 1.0


class A100_SXM_80GBx1_HighAccuracy(A100_SXM_80GBx8_HighAccuracy):
    server_target_qps = 1575


class A100_SXM_80GBx8_Triton(A100_SXM_80GBx8):
    server_target_qps = 22400
    use_triton = True


class A100_SXM_80GBx8_TritonUnified(A100_SXM_80GBx8):
    server_target_qps = 22400
    use_triton = True


class A100_SXM_80GBx8_HighAccuracy_Triton(A100_SXM_80GBx8_HighAccuracy):
    gpu_batch_size = {'bert': 64}
    server_target_qps = 11205
    use_triton = True


class A100_SXM_80GBx8_HighAccuracy_TritonUnified(A100_SXM_80GBx8_HighAccuracy):
    gpu_batch_size = {'bert': 64}
    server_target_qps = 11205
    use_triton = True


class A100_SXM_80GBx8_MaxQ(A100_SXM_80GBx8):
    server_target_qps = 21500
    power_limit = 275


class A100_SXM_80GBx8_HighAccuracy_MaxQ(A100_SXM_80GBx8_MaxQ):
    gpu_batch_size = {'bert': 24}
    precision = "fp16"
    server_target_qps = 10000


class A100_SXM_80GBx8_Triton_MaxQ(A100_SXM_80GBx8_MaxQ):
    server_target_qps = 22455
    use_triton = True


class A100_SXM_80GBx8_TritonUnified_MaxQ(A100_SXM_80GBx8_MaxQ):
    server_target_qps = 22455
    use_triton = True


class A100_SXM_80GBx8_HighAccuracy_Triton_MaxQ(A100_SXM_80GBx8_HighAccuracy_MaxQ):
    gpu_batch_size = {'bert': 48}
    server_target_qps = 11205
    use_triton = True


class A100_SXM_80GBx8_HighAccuracy_TritonUnified_MaxQ(A100_SXM_80GBx8_HighAccuracy_MaxQ):
    gpu_batch_size = {'bert': 48}
    server_target_qps = 11205
    use_triton = True


class A100_SXM_80GB_aarch64x8(ServerGPUBaseConfig):
    active_sms = 60
    gpu_batch_size = {'bert': 48}
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 25400
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True


class A100_SXM_80GB_aarch64x8_MaxQ(A100_SXM_80GB_aarch64x8):
    server_target_qps = 21000
    power_limit = 300


class A100_SXM_80GB_aarch64x8_HighAccuracy(A100_SXM_80GB_aarch64x8):
    gpu_batch_size = {'bert': 24}
    precision = "fp16"
    server_target_qps = 12300


class A100_SXM_80GB_aarch64x8_HighAccuracy_MaxQ(A100_SXM_80GB_aarch64x8_HighAccuracy):
    soft_drop = 0.995
    power_limit = 300
    server_target_qps = 8800


class A100_SXM_80GB_aarch64x8_Triton(A100_SXM_80GB_aarch64x8):
    server_target_qps = 22000
    use_triton = True


class A100_SXM_80GB_aarch64x8_TritonUnified(A100_SXM_80GB_aarch64x8):
    server_target_qps = 22000
    use_triton = True


class A100_SXM_80GB_aarch64x8_HighAccuracy_Triton(A100_SXM_80GB_aarch64x8_HighAccuracy):
    gpu_batch_size = {'bert': 64}
    server_target_qps = 12000
    use_triton = True


class A100_SXM_80GB_aarch64x8_HighAccuracy_TritonUnified(A100_SXM_80GB_aarch64x8_HighAccuracy):
    gpu_batch_size = {'bert': 64}
    server_target_qps = 12000
    use_triton = True


class A100_SXM_80GB_aarch64_MIG_1x1g10gb(ServerGPUBaseConfig):
    gpu_inference_streams = 1
    active_sms = 100
    gpu_batch_size = {'bert': 16}
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 0
    server_target_qps = 320
    soft_drop = 0.99
    use_graphs = False
    use_small_tile_gemm_plugin = True


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Hetero(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    pass


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    precision = "fp16"
    server_target_qps = 220


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Hetero_HighAccuracy(A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy):
    server_target_qps = 220


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Triton(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    server_target_qps = 320
    use_triton = True


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_TritonUnified(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    server_target_qps = 320
    use_triton = True


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy_Triton(A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy):
    precision = "fp16"
    server_target_qps = 220
    use_triton = True


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy_TritonUnified(A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy):
    precision = "fp16"
    server_target_qps = 220
    use_triton = True
