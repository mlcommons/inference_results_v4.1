# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_L40S_PCIe_48GBx4(ServerGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_L40S_PCIe_48GBx4
    gpu_batch_size = {'mixtral-8x7b': 192}
    precision = "fp16"
    use_fp8 = True
    enable_sort = False
    server_target_qps = 8
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    tensor_parallelism = 2  #NVIDIA README says TP=2 for 40GB and lower GPU memory
    pipeline_parallelism = 1
    kvcache_free_gpu_mem_frac = 0.90
    #min_duration = 2400000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_H100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_H100_PCIe_80GBx4
    gpu_batch_size = {'mixtral-8x7b': 1024}
    precision = "fp16"
    use_fp8 = True
    enable_sort = False
    server_target_qps = 18.4
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    tensor_parallelism = 1 #NVIDIA README says TP=1 for 80GB and higher GPU memory
    pipeline_parallelism = 1
    kvcache_free_gpu_mem_frac = 0.90
    #min_duration = 2400000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_PROLIANT_DL380A_H100_NVL_94GBX4(ServerGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_H100_NVL_94GBx4
    gpu_batch_size = {'mixtral-8x7b': 1024}
    precision = "fp16"
    use_fp8 = True
    enable_sort = False
    server_target_qps = 20 * 4
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    tensor_parallelism = 1 #NVIDIA README says TP=1 for 80GB and higher GPU memory
    pipeline_parallelism = 1
    kvcache_free_gpu_mem_frac = 0.90
    #min_duration = 2400000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_CRAY_XD670_H100_SXM_80GBX8(H100_SXM_80GBx8):
    system = KnownSystem.HPE_Cray_XD670_H100_SXM_80GBx8
    gpu_batch_size = {'mixtral-8x7b': 1024}
    precision = "fp16"
    use_fp8 = True
    server_target_qps = 40*8 #45 * 8 #43.5 * 8
    enable_sort = False
    vboost_slider = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1 #2
    tensor_parallelism = 1 #NVIDIA README says TP=1 for 80GB and higher GPU memory
    pipeline_parallelism = 1
    kvcache_free_gpu_mem_frac = 0.90 #0.90
    min_duration = 2400000
