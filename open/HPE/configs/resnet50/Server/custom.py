# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_L40S_PCIe_48GBx4(ServerGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_L40S_PCIe_48GBx4
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = {'resnet50': 64} 
    gpu_copy_streams = 4*4
    gpu_inference_streams = 2*4
    server_target_qps = 44000*4
    use_cuda_thread_per_device = False
    use_graphs = False

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_L40S_PCIe_48GBx4_Triton(HPE_ProLiant_DL380a_L40S_PCIe_48GBx4):
    use_triton = True

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_H100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_H100_PCIe_80GBx4
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = {'resnet50': 128}
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 51000*4
    use_cuda_thread_per_device = True
    use_batcher_thread_per_device = True
    use_graphs = True
    #start_from_device = True

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_PROLIANT_DL380A_H100_NVL_94GBX4(ServerGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_H100_NVL_94GBx4
    use_deque_limit = True
    deque_timeout_usec = 3000
    gpu_batch_size = {'resnet50': 256}
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    server_target_qps = 60000 * 4
    use_cuda_thread_per_device = True
    #use_graphs = True
    #start_from_device = True

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_PROLIANT_DL380A_H100_NVL_94GBX4_Triton(HPE_PROLIANT_DL380A_H100_NVL_94GBX4):
    use_triton = True

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_CRAY_XD670_H100_SXM_80GBX8(H100_SXM_80GBx8):
    system = KnownSystem.HPE_Cray_XD670_H100_SXM_80GBx8
    start_from_device = True
    #run_infer_on_copy_streams = True
    use_deque_limit = True
    deque_timeout_usec = 4182
    use_cuda_thread_per_device = True
    use_graphs = True
    gpu_batch_size = {'resnet50': 384}
    gpu_copy_streams = 5
    gpu_inference_streams = 1
    server_target_qps = 620000

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_CRAY_XD670_H100_SXM_80GBX8_Triton(HPE_CRAY_XD670_H100_SXM_80GBX8):
    use_triton = True

"""
    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    gpu_batch_size: Dict = {}
    input_dtype: str = ''
    input_format: str = ''
    map_path: str = ''
    precision: str = ''
    tensor_path: str = ''

    # Optional fields:
    active_sms: int = 0
    assume_contiguous: bool = False
    buffer_manager_thread_count: int = 0
    cache_file: str = ''
    complete_threads: int = 0
    deque_timeout_usec: int = 0
    disable_beta1_smallk: bool = False
    energy_aware_kernels: bool = False
    gpu_copy_streams: int = 0
    gpu_inference_streams: int = 0
    instance_group_count: int = 0
    model_path: str = ''
    numa_config: str = ''
    performance_sample_count_override: int = 0
    preferred_batch_size: str = ''
    request_timeout_usec: int = 0
    run_infer_on_copy_streams: bool = False
    schedule_rng_seed: int = 0
    server_num_issue_query_threads: int = 0
    server_target_latency_ns: int = 0
    server_target_latency_percentile: float = 0.0
    server_target_qps: float = 0.0
    server_target_qps_adj_factor: float = 0.0
    use_batcher_thread_per_device: bool = False
    use_cuda_thread_per_device: bool = False
    use_deque_limit: bool = False
    use_graphs: bool = False
    use_jemalloc: bool = False
    use_same_context: bool = False
    use_spin_wait: bool = False
    vboost_slider: int = 0
    verbose_glog: int = 0
    warmup_duration: float = 0.0
    workspace_size: int = 0
"""
