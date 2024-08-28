# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_L40S_PCIe_48GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_L40S_PCIe_48GBx4
    gpu_batch_size = {'retinanet': 2}
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    offline_expected_qps = 950*4
    run_infer_on_copy_streams = False
    workspace_size = 60000000000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_H100_PCIe_80GBx4(H100_PCIe_80GBx1):
    system = KnownSystem.HPE_ProLiant_DL380a_H100_PCIe_80GBx4
    gpu_batch_size = {'retinanet': 16}
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 1250*4 #1150*4
    run_infer_on_copy_streams = False
    workspace_size = 60000000000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_PROLIANT_DL380A_H100_NVL_94GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_H100_NVL_94GBx4
    gpu_batch_size = {'retinanet': 56} #48
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 1850 * 4 #1300 * 4 #1150 * 4
    run_infer_on_copy_streams = False
    workspace_size = 60000000000

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_PROLIANT_DL380A_H100_NVL_94GBX4_Triton(HPE_PROLIANT_DL380A_H100_NVL_94GBX4):
    use_triton = True

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_CRAY_XD670_H100_SXM_80GBX8(H100_SXM_80GBx8):
    system = KnownSystem.HPE_Cray_XD670_H100_SXM_80GBx8
    gpu_batch_size = {'retinanet': 48}
    start_from_device = True
    offline_expected_qps = 2000 * 8 #1900 * 8 #2200 * 8 #1650 * 8 #1750 * 8 #1800 * 8 #1700 * 8

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
    nms_type: str = ''
    numa_config: str = ''
    offline_expected_qps: float = 0.0
    performance_sample_count_override: int = 0
    preferred_batch_size: str = ''
    request_timeout_usec: int = 0
    run_infer_on_copy_streams: bool = False
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

