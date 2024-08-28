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

ParentConfig = import_module("configs.mixtral-8x7b")
GPUBaseConfig = ParentConfig.GPUBaseConfig


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    tensor_parallelism = 2
    pipeline_parallelism = 1
    precision = "fp16"
    enable_sort = False
    kvcache_free_gpu_mem_frac = 0.90
    min_duration = 2400000


class GH200_144GB_aarch64x1(ServerGPUBaseConfig):
    gpu_batch_size = {'mixtral-8x7b': 1024}
    use_fp8 = True
    enable_sort = False
    tensor_parallelism = 1
    server_target_qps = 40


class H100_SXM_80GBx1(ServerGPUBaseConfig):
    gpu_batch_size = {'mixtral-8x7b': 896}
    use_fp8 = True
    server_target_qps = 43.5
    enable_sort = False
    tensor_parallelism = 1
    vboost_slider = 1


class H100_SXM_80GBx8(H100_SXM_80GBx1):
    server_target_qps = 43.5 * 8


class H200_SXM_141GBx1(ServerGPUBaseConfig):
    # gpu_batch_size = {'mixtral-8x7b': 3072}
    gpu_batch_size = {'mixtral-8x7b': 1200}
    use_fp8 = True
    max_num_tokens = 8192
    server_target_qps = 49
    enable_sort = False
    tensor_parallelism = 1
    vboost_slider = 1


class H200_SXM_141GBx8(H200_SXM_141GBx1):
    server_target_qps = 49 * 8


class H200_SXM_141GB_CTSx1(ServerGPUBaseConfig):
    gpu_batch_size = {'mixtral-8x7b': 3072}
    use_fp8 = True
    server_target_qps = 38
    enable_sort = False
    tensor_parallelism = 1
    vboost_slider = 1


class H200_SXM_141GB_CTSx8(H200_SXM_141GB_CTSx1):
    server_target_qps = 38 * 8
