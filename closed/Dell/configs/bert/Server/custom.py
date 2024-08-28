from . import * 

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_H100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.R760xa_H100_PCIe_80GBx4
    gpu_batch_size = {'bert': 276}
    graphs_max_seqlen: int = 200
    server_target_qps: int = 17882
    server_num_issue_query_threads: int = 1
    soft_drop: float = 0.99
    use_graphs: bool = False
    use_small_tile_gemm_plugin: bool = False
    workspace_size: int = 7516192768



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_L40Sx4(ServerGPUBaseConfig):
    system = KnownSystem.R760xa_L40Sx4
    gpu_batch_size = {'bert': 64}
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    server_target_qps = 13850
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    workspace_size = 7000000000000
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9640_H100_SXM_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.XE9640_H100_SXM_80GBx4
    use_small_tile_gemm_plugin = False
    use_graphs = False
    gpu_batch_size = {'bert': 120}
    gpu_copy_streams = 3
    gpu_inference_streams = 2
    server_target_qps = 28343
    server_num_issue_query_threads = 1
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9640_H100_SXM_80GBX4_HighAccuracy(XE9640_H100_SXM_80GBX4):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = {'bert': 145}
    server_target_qps = 24850


ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760XA_H100NVL_PCIE_94GBX4(ServerGPUBaseConfig):
    system = KnownSystem.R760xa_H100NVL_PCIe_94GBx4
    use_small_tile_gemm_plugin = False
    use_graphs = False
    gpu_batch_size = {'bert': 1320}
    server_target_qps = 20100
    server_num_issue_query_threads = 1
    #workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R760XA_H100NVL_PCIE_94GBX4_HighAccuracy(R760XA_H100NVL_PCIE_94GBX4):
    precision = "fp16"
    use_fp8 = True
    use_small_tile_gemm_plugin = False
    use_graphs = False
    gpu_batch_size = {'bert': 1280}
    server_target_qps = 17600
    server_num_issue_query_threads = 1
    #workspace_size = 7516192768



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.XE8640_H100_SXM_80GBx4
    use_small_tile_gemm_plugin = False
    use_graphs = False
    gpu_batch_size = {'bert': 204}
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    server_target_qps = 28664 
    server_num_issue_query_threads = 1
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4_HighAccuracy(XE8640_H100_SXM_80GBx4):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = {'bert': 170}
    server_target_qps =25390


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H200_SXM_141GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680_H200_SXM_141GBx8
    use_small_tile_gemm_plugin = False
    use_graphs = False
    gpu_batch_size = {'bert': 128}
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 7260 * 8
    server_num_issue_query_threads = 1
    workspace_size = 7516192768
    vboost_slider = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_H200_SXM_141GBX8_HighAccuracy(XE9680_H200_SXM_141GBX8):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = {'bert': 512}
    gpu_inference_streams = 1
    server_target_qps = 6400 * 8

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    use_small_tile_gemm_plugin = False
    use_graphs = False
    gpu_batch_size = {'bert': 128}
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 7000 * 8
    server_num_issue_query_threads = 1
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8_HighAccuracy(XE9680_H100_SXM_80GBX8):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = {'bert': 512}
    gpu_inference_streams = 1
    server_target_qps = 6200 * 8

