from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_L40Sx4(OfflineGPUBaseConfig):
    system = KnownSystem.R760xa_L40Sx4
    use_small_tile_gemm_plugin = True
    gpu_batch_size = {'bert': 32}
    offline_expected_qps = 13600
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_H100_PCIe_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.R760xa_H100_PCIe_80GBx4
    gpu_batch_size = {'bert': 1280}
    offline_expected_qps: float = 25000
    use_small_tile_gemm_plugin: bool = False
    workspace_size: int = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9640_H100_SXM_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.XE9640_H100_SXM_80GBx4
    use_small_tile_gemm_plugin = False
    gpu_batch_size = {'bert': 1313}
    gpu_copy_streams = 2
    gpu_inference_streams = 3
    offline_expected_qps = 39394

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9640_H100_SXM_80GBX4_HighAccuracy(XE9640_H100_SXM_80GBX4):
    precision = "fp16"
    offline_expected_qps = 34333
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = {'bert': 1369}
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760XA_H100NVL_PCIE_94GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.R760xa_H100NVL_PCIe_94GBx4
    use_small_tile_gemm_plugin = False
    gpu_batch_size = {'bert': 1320}
    offline_expected_qps = 25000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R760XA_H100NVL_PCIE_94GBX4_HighAccuracy(R760XA_H100NVL_PCIE_94GBX4):
    precision = "fp16"
    use_fp8 = True
    offline_expected_qps = 22500
    use_graphs = False
    gpu_batch_size = {'bert': 1096}



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.XE8640_H100_SXM_80GBx4
    use_small_tile_gemm_plugin = False
    gpu_batch_size = {'bert': 1280}
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    offline_expected_qps = 9400*4
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4_HighAccuracy(XE8640_H100_SXM_80GBx4):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = {'bert': 1024}
    offline_expected_qps = 8200*4


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H200_SXM_141GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H200_SXM_141GBx8
    use_small_tile_gemm_plugin = False
    gpu_batch_size = {'bert': 1280}
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    offline_expected_qps = 9400 * 8
    workspace_size = 128000000000
    vboost_slider = 1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_H200_SXM_141GBX8_HighAccuracy(XE9680_H200_SXM_141GBX8):
    gpu_batch_size = {'bert': 1024}
    precision = "fp16"
    use_fp8 = True
    offline_expected_qps = 8200 * 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    use_small_tile_gemm_plugin = False
    gpu_batch_size = {'bert': 1280}
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    offline_expected_qps = 9400 * 8
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8_HighAccuracy(XE9680_H100_SXM_80GBX8):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = {'bert': 1024}
    offline_expected_qps = 8200 * 8

