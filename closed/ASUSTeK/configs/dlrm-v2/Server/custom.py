# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ESC8000A_E12_H100X8(H100_PCIe_80GBx8):
    system = KnownSystem.ESC8000A_E12_H100x8
    server_target_qps = 170000


    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ESC8000A_E12_H100X8_HighAccuracy(ESC8000A_E12_H100X8):
    interaction_op_precision = 'fp16'
    server_target_qps = 170000


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
    #server_target_qps = 500000
    server_target_qps = 64500 * 8
    numa_config = "0-3:0-63,128-191&4-7:64-127,192-255"

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ESC_N8_E11_H100x8_HighAccuracy(H100_SXM_80GBx8_HighAccuracy):
    system = KnownSystem.ESC_N8_E11_H100x8
    numa_config = "0-3:0-63,128-191&4-7:64-127,192-255"
  


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ESC4000A_E12_4XH100(ServerGPUBaseConfig):
    system = KnownSystem.ESC4000A_E12_4XH100
    gpu_batch_size = {'dlrm-v2': 102400}
    embedding_weights_on_gpu_part: float = 1.0
    server_target_qps = 129375.0
    server_num_issue_query_threads = 4


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ESC4000A_E12_4XH100_HighAccuracy(ESC4000A_E12_4XH100):
    server_target_qps = 100000.0
    interaction_op_precision = 'fp16'

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
    bot_mlp_precision: str = ''
    buffer_manager_thread_count: int = 0
    cache_file: str = ''
    check_contiguity: bool = False
    coalesced_tensor: bool = False
    complete_threads: int = 0
    embedding_weights_on_gpu_part: float = 0.0
    embeddings_path: str = ''
    embeddings_precision: str = ''
    final_linear_precision: str = ''
    gather_kernel_buffer_threshold: int = 0
    gpu_copy_streams: int = 0
    gpu_inference_streams: int = 0
    gpu_num_bundles: int = 0
    instance_group_count: int = 0
    interaction_op_precision: str = ''
    max_pairs_per_staging_thread: int = 0
    max_queue_delay_usec: int = 0
    model_path: str = ''
    num_concurrent_batchers: int = 0
    num_concurrent_issuers: int = 0
    num_staging_batches: int = 0
    num_staging_threads: int = 0
    numa_config: str = ''
    output_pinned_memory: bool = False
    performance_sample_count_override: int = 0
    preferred_batch_size: str = ''
    qsl_numa_override: str = ''
    request_timeout_usec: int = 0
    run_infer_on_copy_streams: bool = False
    sample_partition_path: str = ''
    schedule_rng_seed: int = 0
    server_num_issue_query_threads: int = 0
    server_target_latency_ns: int = 0
    server_target_latency_percentile: float = 0.0
    server_target_qps: float = 0.0
    server_target_qps_adj_factor: float = 0.0
    top_mlp_precision: str = ''
    use_batcher_thread_per_device: bool = False
    use_concurrent_harness: bool = False
    use_graphs: bool = False
    use_jemalloc: bool = False
    use_spin_wait: bool = False
    vboost_slider: int = 0
    verbose_glog: int = 0
    warmup_duration: float = 0.0
    workspace_size: int = 0


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ESC4000A_E12_4XH100_HighAccuracy_Triton(ESC4000A_E12_4XH100_HighAccuracy):
    use_triton = True


