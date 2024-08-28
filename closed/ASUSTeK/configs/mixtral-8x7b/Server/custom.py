# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ESC8000A_E12_H100X8(ServerGPUBaseConfig):
    system = KnownSystem.ESC8000A_E12_H100x8

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    gpu_batch_size: Dict = {}
    input_dtype: str = ''
    input_format: str = ''
    precision: str = ''
    tensor_path: str = ''

    # Optional fields:
    active_sms: int = 0
    buffer_manager_thread_count: int = 0
    cache_file: str = ''
    coalesced_tensor: bool = False
    enable_sort: bool = False
    gpu_copy_streams: int = 0
    gpu_inference_streams: int = 0
    gpu_rank_map: str = ''
    instance_group_count: int = 0
    kvcache_free_gpu_mem_frac: float = 0.0
    llm_gen_config_path: str = ''
    max_num_tokens: int = 0
    model_path: str = ''
    num_sort_segments: int = 0
    numa_config: str = ''
    performance_sample_count_override: int = 0
    pipeline_parallelism: int = 0
    preferred_batch_size: str = ''
    request_timeout_usec: int = 0
    run_infer_on_copy_streams: bool = False
    schedule_rng_seed: int = 0
    server_num_issue_query_threads: int = 0
    server_target_latency_ns: int = 0
    server_target_latency_percentile: float = 0.0
    server_target_qps: float = 0.0
    server_target_qps_adj_factor: float = 0.0
    tensor_parallelism: int = 0
    trtllm_batch_sched_policy: str = ''
    trtllm_batching_mode: str = ''
    use_fp8: bool = False
    use_graphs: bool = False
    use_jemalloc: bool = False
    use_spin_wait: bool = False
    use_token_latencies: bool = False
    vboost_slider: int = 0
    verbose_glog: int = 0
    workspace_size: int = 0


