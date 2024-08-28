# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ESC8000A_E12_H100X8(H100_PCIe_80GBx8):
    system = KnownSystem.ESC8000A_E12_H100x8

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ESC8000A_E12_H100X8_HighAccuracy(ESC8000A_E12_H100X8):
    precision = "fp16"
    #offline_expected_qps = ESC8000A_E12_H100X8.offline_expected_qps / 2
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = {'bert': 1024}

    offline_expected_qps = 39000



@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class ESC8000A_E12_H100X8_Triton(ESC8000A_E12_H100X8):
    use_triton = True

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ESC8000A_E12_H100X8_HighAccuracy_Triton(ESC8000A_E12_H100X8_HighAccuracy):
    use_triton = True




@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ESC_N8_E11_H100x8(H100_SXM_80GBx8):
    system = KnownSystem.ESC_N8_E11_H100x8
    use_small_tile_gemm_plugin = False
    gpu_batch_size = {'bert' : 1280}
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    workspace_size = 7516192768
    offline_expected_qps = 9400 * 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ESC_N8_E11_H100x8_HighAccuracy(H100_SXM_80GBx8_HighAccuracy):
    system = KnownSystem.ESC_N8_E11_H100x8
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = {'bert' : 1024}
    offline_expected_qps = 8200 * 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ESC4000A_E12_4XH100(OfflineGPUBaseConfig):
    system = KnownSystem.ESC4000A_E12_4XH100
    gpu_batch_size = {'bert': 512}
    offline_expected_qps = 36850.0
    workspace_size = 7516192768
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    use_small_tile_gemm_plugin = False

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ESC4000A_E12_4XH100_HighAccuracy(ESC4000A_E12_4XH100):
    precision = "fp16"
    use_fp8 = True
    gpu_batch_size = {'bert': 256}
    offline_expected_qps = 34450.0
    gpu_inference_streams = 1
    gpu_copy_streams = 2
    use_graphs = False

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class ESC4000A_E12_4XH100_Triton(ESC4000A_E12_4XH100):
    use_triton = True

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    gpu_batch_size: {}
    input_dtype: str = ''
    input_format: str = ''
    precision: str = ''
    tensor_path: str = ''

    # Optional fields:
    active_sms: int = 0
    batch_triton_requests: bool = False
    bert_opt_seqlen: int = 0
    buffer_manager_thread_count: int = 0
    cache_file: str = ''
    coalesced_tensor: bool = False
    deque_timeout_usec: int = 0
    energy_aware_kernels: bool = False
    gather_kernel_buffer_threshold: int = 0
    gemm_plugin_fairshare_cache_size: int = 0
    gpu_copy_streams: int = 0
    gpu_inference_streams: int = 0
    graph_specs: str = ''
    graphs_max_seqlen: int = 0
    instance_group_count: int = 0
    max_queue_delay_usec: int = 0
    model_path: str = ''
    num_concurrent_batchers: int = 0
    num_concurrent_issuers: int = 0
    numa_config: str = ''
    offline_expected_qps: float = 0.0
    output_pinned_memory: bool = False
    performance_sample_count_override: int = 0
    preferred_batch_size: str = ''
    request_timeout_usec: int = 0
    run_infer_on_copy_streams: bool = False
    soft_drop: float = 0.0
    use_concurrent_harness: bool = False
    use_fp8: bool = False
    use_graphs: bool = False
    use_jemalloc: bool = False
    use_small_tile_gemm_plugin: bool = False
    use_spin_wait: bool = False
    vboost_slider: int = 0
    verbose_glog: int = 0
    workspace_size: int = 0


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ESC4000A_E12_4XH100_HighAccuracy_Triton(ESC4000A_E12_4XH100_HighAccuracy):
    use_triton = True
