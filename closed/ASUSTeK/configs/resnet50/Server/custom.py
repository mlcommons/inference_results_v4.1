# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class ESC8000A_E12_H100X8(H100_PCIe_80GBx8):
    system = KnownSystem.ESC8000A_E12_H100x8
    server_target_qps = 410000


    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class ESC8000A_E12_H100X8_Triton(ESC8000A_E12_H100X8):
    use_triton = True

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):




@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class ESC_N8_E11_H100x8(H100_SXM_80GBx8):
    system = KnownSystem.ESC_N8_E11_H100x8
    #server_target_qps = 580000
    use_deque_limit = True
    use_cuda_thread_per_device = True
    use_graphs = True
    numa_config = "0-3:0-63,128-191&4-7:64-127,192-255"
    gpu_batch_size = {'resnet50': 391}
    server_target_qps = 630000
    gpu_inference_streams = 1
    gpu_copy_streams = 5
    deque_timeout_usec = 4182



@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class ESC4000A_E12_4XH100(ServerGPUBaseConfig):
    system = KnownSystem.ESC4000A_E12_4XH100
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = {'resnet50':128}
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 233250.0
    use_cuda_thread_per_device = True
    use_batcher_thread_per_device = True
    use_graphs = True

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class ESC4000A_E12_4XH100_Triton(ESC4000A_E12_4XH100):
    use_triton = True

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    gpu_batch_size: {}
    input_dtype: str = ''
    input_format: str = ''
    map_path: str = ''
    precision: str = ''
    tensor_path: str = ''

    # Optional fields:
    active_sms: int = 0
    assume_contiguous: bool = False
    batch_triton_requests: bool = False
    buffer_manager_thread_count: int = 0
    cache_file: str = ''
    complete_threads: int = 0
    deque_timeout_usec: int = 0
    disable_beta1_smallk: bool = False
    energy_aware_kernels: bool = False
    gather_kernel_buffer_threshold: int = 0
    gpu_copy_streams: int = 0
    gpu_inference_streams: int = 0
    instance_group_count: int = 0
    max_queue_delay_usec: int = 0
    model_path: str = ''
    num_concurrent_batchers: int = 0
    num_concurrent_issuers: int = 0
    numa_config: str = ''
    output_pinned_memory: bool = False
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
    use_concurrent_harness: bool = False
    use_cuda_thread_per_device: bool = False
    use_deque_limit: bool = False
    use_graphs: bool = False
    use_jemalloc: bool = False
    use_same_context: bool = False
    use_spin_wait: bool = False
    vboost_slider: int = 0
    verbose_glog: int = 0
    warmup_duration: float = 0.0
