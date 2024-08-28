# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GIGABYTE_G593_SD1_H200_SXM_141GBX8(ServerGPUBaseConfig):
    system = KnownSystem.GIGABYTE_G593_SD1_H200_SXM_141GBx8
    use_small_tile_gemm_plugin = False
    use_graphs = False
    gpu_batch_size = {'bert': 128}
    server_num_issue_query_threads = 1
    workspace_size = 7516192768
    vboost_slider = 1
    gpu_inference_streams = 2
    gpu_copy_streams = 4
    #server_target_qps = H200_SXM_141GBx1.server_target_qps * 8
    server_target_qps = 7260 * 8

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class GIGABYTE_G593_SD1_H200_SXM_141GBX8_HighAccuracy(GIGABYTE_G593_SD1_H200_SXM_141GBX8):
    #precision = "fp16"
    #server_target_qps = GIGABYTE_G593_SD3_H200_SXM_141GBX8.server_target_qps / 2
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = {'bert': 512}
    gpu_inference_streams = 1
    server_target_qps = 51200 # 6200 * 8
    use_small_tile_gemm_plugin = False
    gpu_copy_streams = 6
    server_num_issue_query_threads = 1
    workspace_size = 7516192768




