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


class GH200_96GB_aarch64x1(ServerGPUBaseConfig):
    use_deque_limit = True
    deque_timeout_usec = 3000
    gpu_batch_size = {'resnet50': 256}
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    server_target_qps = 77000
    use_cuda_thread_per_device = True
    use_graphs = True
    start_from_device = True


class L4x1(ServerGPUBaseConfig):
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = {'resnet50': 16}
    gpu_copy_streams = 9
    gpu_inference_streams = 2
    server_target_qps = 12200
    use_cuda_thread_per_device = True
    # use_graphs = True


class L40x1(ServerGPUBaseConfig):
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = {'resnet50': 64}
    gpu_copy_streams = 9
    gpu_inference_streams = 2
    server_target_qps = 16000
    use_cuda_thread_per_device = True
    use_graphs = True


class H200_SXM_141GBx1(ServerGPUBaseConfig):
    use_deque_limit = True
    deque_timeout_usec = 3000
    gpu_batch_size = {'resnet50': 256}
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    server_target_qps = 79000
    use_cuda_thread_per_device = True
    use_graphs = True
    start_from_device = True


class H200_SXM_141GBx8(H200_SXM_141GBx1):
    gpu_batch_size = {'resnet50': 256}
    gpu_copy_streams = 4
    gpu_inference_streams = 7
    server_target_qps = H200_SXM_141GBx1.server_target_qps * 8


class H200_SXM_141GBx8_MaxQ(H200_SXM_141GBx8):
    power_limit = 300
    server_target_qps = 50000 * 8


class H100_SXM_80GBx1(ServerGPUBaseConfig):
    use_deque_limit = True
    deque_timeout_usec = 3000
    gpu_batch_size = {'resnet50': 256}
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    server_target_qps = 73000
    use_cuda_thread_per_device = True
    use_graphs = True
    start_from_device = True


class H100_SXM_80GBx1_Triton(H100_SXM_80GBx1):
    deque_timeout_usec = 500
    gpu_batch_size = {'resnet50': 64}
    gpu_inference_streams = 5
    server_target_qps = 55000
    use_graphs = False
    use_triton = True
    batch_triton_requests = True
    max_queue_delay_usec = 1000
    request_timeout_usec = 2000


class H100_SXM_80GBx8(H100_SXM_80GBx1):
    gpu_batch_size = {'resnet50': 256}
    gpu_copy_streams = 4
    gpu_inference_streams = 7
    server_target_qps = H100_SXM_80GBx1.server_target_qps * 8


class H100_SXM_80GBx8_MaxQ(H100_SXM_80GBx8):
    power_limit = 300
    server_target_qps = 50000 * 8


class H100_SXM_80GBx8_Triton(H100_SXM_80GBx1_Triton):
    gpu_batch_size = {'resnet50': 128}
    server_target_qps = 55000 * 8


class H100_PCIe_80GBx1(ServerGPUBaseConfig):
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = {'resnet50': 128}
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 47000
    use_cuda_thread_per_device = True
    use_batcher_thread_per_device = True
    use_graphs = True


class H100_PCIe_80GBx1_Triton(H100_PCIe_80GBx1):
    deque_timeout_usec = 500
    gpu_batch_size = {'resnet50': 128}
    gpu_inference_streams = 5
    server_target_qps = 40000
    use_graphs = False
    use_triton = True
    batch_triton_requests = True
    max_queue_delay_usec = 1000
    request_timeout_usec = 2000


class H100_PCIe_80GBx8(H100_PCIe_80GBx1):
    server_target_qps = 46000 * 8
    numa_config = "3:0-15&2:16-31&1:32-47&0:48-63&7:64-79&6:80-95&5:96-111&4:112-127"


class H100_PCIe_80GBx8_MaxQ(H100_PCIe_80GBx8):
    power_limit = 210
    server_target_qps = 240000
    numa_config = "3:0-7&2:8-15&1:16-23&0:24-31&7:32-39&6:40-47&5:48-55&4:56-63"


class H100_PCIe_80GBx8_Triton(H100_PCIe_80GBx1_Triton):
    gpu_batch_size = {'resnet50': 128}
    server_target_qps = 240000


class H100_PCIe_80GB_aarch64x1(ServerGPUBaseConfig):
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = {'resnet50': 128}
    gpu_copy_streams = 4
    gpu_inference_streams = 7
    server_target_qps = 45000
    use_cuda_thread_per_device = True
    use_graphs = True


class H100_PCIe_80GB_aarch64x4(H100_PCIe_80GB_aarch64x1):
    gpu_batch_size = {'resnet50': 128}
    server_target_qps = 250000


class A100_PCIe_80GBx8(ServerGPUBaseConfig):
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = {'resnet50': 64}
    gpu_copy_streams = 9
    gpu_inference_streams = 2
    server_target_qps = 236000
    use_cuda_thread_per_device = True
    use_graphs = True


class A100_PCIe_80GBx8_Triton(A100_PCIe_80GBx8):
    deque_timeout_usec = 500
    gpu_batch_size = {'resnet50': 64}
    gpu_inference_streams = 5
    server_target_qps = 230000
    use_graphs = False
    use_triton = True
    batch_triton_requests = True
    max_queue_delay_usec = 1000
    request_timeout_usec = 2000
    numa_config = "3:0-15,128-143&2:16-31,144-159&1:32-47,160-175&0:48-63,176-191&7:64-79,192-207&6:80-95,208-223&5:96-111,224-239&4:112-127,240-255"


class A100_PCIe_80GBx8_TritonUnified(A100_PCIe_80GBx8_Triton):
    batch_triton_requests = True
    deque_timeout_usec = 500
    gpu_batch_size = {'resnet50': 64}
    gpu_inference_streams = 5
    server_target_qps = 220000
    numa_config = None
    use_graphs = False
    use_triton = True
    max_queue_delay_usec = 1000
    request_timeout_usec = 2000


class A100_PCIe_80GBx8_MaxQ(A100_PCIe_80GBx8):
    numa_config = "3:0-7&2:8-15&1:16-23&0:24-31&7:32-39&6:40-47&5:48-55&4:56-63"
    gpu_batch_size = {'resnet50': 128}
    gpu_inference_streams = 3
    server_target_qps = 203500
    power_limit = 200


class A100_PCIe_80GBx8_Triton_MaxQ(A100_PCIe_80GBx8_MaxQ):
    server_target_qps = 130000
    use_graphs = False
    power_limit = 175
    use_triton = True


class A100_PCIe_80GBx8_TritonUnified_MaxQ(A100_PCIe_80GBx8_MaxQ):
    server_target_qps = 130000
    use_graphs = False
    power_limit = 175
    use_triton = True


class A100_PCIe_80GB_aarch64x1(ServerGPUBaseConfig):
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = {'resnet50': 64}
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 26000
    use_cuda_thread_per_device = True
    use_graphs = True


class A100_PCIe_80GB_aarch64x4(A100_PCIe_80GB_aarch64x1):
    gpu_batch_size = {'resnet50': 128}
    gpu_copy_streams = 4
    gpu_inference_streams = 5
    server_target_qps = 104000


class A100_PCIe_80GB_aarch64x4_Triton(A100_PCIe_80GB_aarch64x4):
    num_concurrent_batchers = 6
    num_concurrent_issuers = 6
    server_target_qps = 93200
    use_triton = True


class A100_PCIe_80GB_aarch64x4_TritonUnified(A100_PCIe_80GB_aarch64x4):
    num_concurrent_batchers = 6
    num_concurrent_issuers = 6
    server_target_qps = 93200
    use_triton = True


class A100_PCIe_80GB_aarch64x4_MaxQ(A100_PCIe_80GB_aarch64x4):
    server_target_qps = 92500
    power_limit = 175


class A100_PCIe_80GB_MIG_1x1g10gb(ServerGPUBaseConfig):
    use_deque_limit = True
    deque_timeout_usec = 1000
    gpu_batch_size = {'resnet50': 8}
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    server_target_qps = 3600
    use_graphs = True


class A100_PCIe_80GB_MIG_1x1g10gb_Hetero(A100_PCIe_80GB_MIG_1x1g10gb):
    pass


class A100_PCIe_80GB_MIG_1x1g10gb_Triton(A100_PCIe_80GB_MIG_1x1g10gb):
    server_target_qps = 3500
    use_graphs = False
    use_triton = True


class A100_PCIe_80GB_MIG_1x1g10gb_TritonUnified(A100_PCIe_80GB_MIG_1x1g10gb):
    server_target_qps = 3500
    use_graphs = False
    use_triton = True


class A100_SXM_80GB_MIG_1x1g10gb(ServerGPUBaseConfig):
    use_deque_limit = True
    deque_timeout_usec = 1000
    gpu_batch_size = {'resnet50': 8}
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    server_target_qps = 3530
    start_from_device = True
    use_graphs = True


class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    server_target_qps = 3600


class A100_SXM_80GB_MIG_1x1g10gb_Triton(A100_SXM_80GB_MIG_1x1g10gb):
    server_target_qps = 3440
    use_graphs = False
    use_triton = True


class A100_SXM_80GB_MIG_1x1g10gb_TritonUnified(A100_SXM_80GB_MIG_1x1g10gb):
    server_target_qps = 3440
    use_graphs = False
    use_triton = True


class A100_SXM_80GBx8(ServerGPUBaseConfig):
    use_deque_limit = True
    deque_timeout_usec = 4000
    gpu_batch_size = {'resnet50': 128}
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 290000
    start_from_device = True
    use_cuda_thread_per_device = True
    use_graphs = True


class A100_SXM_80GBx8_Triton(A100_SXM_80GBx8):
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    instance_group_count = 2
    batch_triton_requests = False
    server_target_qps = 200000
    gather_kernel_buffer_threshold = 0
    gpu_batch_size = {'resnet50': 64}
    num_concurrent_batchers = 8
    num_concurrent_issuers = 8
    request_timeout_usec = 2000
    use_graphs = True
    start_from_device = False
    use_triton = True


class A100_SXM_80GBx8_TritonUnified(A100_SXM_80GBx8):
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    instance_group_count = 2
    batch_triton_requests = False
    server_target_qps = 200000
    gather_kernel_buffer_threshold = 0
    gpu_batch_size = {'resnet50': 64}
    num_concurrent_batchers = 8
    num_concurrent_issuers = 8
    request_timeout_usec = 2000
    use_graphs = True
    start_from_device = False
    use_triton = True


class A100_SXM_80GBx8_MaxQ(A100_SXM_80GBx8):
    server_target_qps = 229000
    power_limit = 225


class A100_SXM_80GBx8_TritonUnified_MaxQ(A100_SXM_80GBx8_MaxQ):
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 200000
    use_graphs = False
    start_from_device = False
    use_triton = True


class A100_SXM_80GB_aarch64_MIG_1x1g10gb(ServerGPUBaseConfig):
    use_deque_limit = True
    deque_timeout_usec = 1000
    gpu_batch_size = {'resnet50': 8}
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    server_target_qps = 3600
    use_graphs = True


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Hetero(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    pass


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Triton(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    server_target_qps = 3500
    use_graphs = False
    use_triton = True


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_TritonUnified(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    server_target_qps = 3500
    use_graphs = False
    use_triton = True


class A100_SXM_80GB_aarch64x8(ServerGPUBaseConfig):
    use_deque_limit = True
    deque_timeout_usec = 4000
    gpu_batch_size = {'resnet50': 128}
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 270000
    use_cuda_thread_per_device = True
    use_graphs = True


class A100_SXM_80GB_aarch64x8_MaxQ(A100_SXM_80GB_aarch64x8):
    server_target_qps = 230000
    power_limit = 250


class A100_SXM_80GB_aarch64x8_Triton(A100_SXM_80GB_aarch64x8):
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    instance_group_count = 2
    batch_triton_requests = False
    server_target_qps = 186000
    gather_kernel_buffer_threshold = 0
    num_concurrent_batchers = 8
    num_concurrent_issuers = 8
    request_timeout_usec = 2000
    use_triton = True


class A100_SXM_80GB_aarch64x8_TritonUnified(A100_SXM_80GB_aarch64x8):
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    instance_group_count = 2
    batch_triton_requests = False
    server_target_qps = 186000
    gather_kernel_buffer_threshold = 0
    num_concurrent_batchers = 8
    num_concurrent_issuers = 8
    request_timeout_usec = 2000
    use_triton = True
