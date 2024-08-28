from . import * 

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9640_H100_SXM_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.XE9640_H100_SXM_80GBx4
    gpu_batch_size = {'clip1': 8 * 2, 'clip2': 8 * 2, 'unet': 8 * 2, 'vae': 8}
    server_target_qps = 6.49
    sdxl_batcher_time_limit = 4
    use_graphs = False  # disable to meet latency constraint for x1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760XA_H100NVL_PCIE_94GBX4(ServerGPUBaseConfig):
    system = KnownSystem.R760xa_H100NVL_PCIe_94GBx4
    gpu_batch_size = {'clip1': 8 * 2, 'clip2': 8 * 2, 'unet': 8 * 2, 'vae': 8}
    server_target_qps = 5 
    sdxl_batcher_time_limit = 3
    use_graphs = False  # disable to meet latency constraint for x1
    vboost_slider = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H200_SXM_141GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680_H200_SXM_141GBx8
    gpu_batch_size = {'clip1': 8 * 2, 'clip2': 8 * 2, 'unet': 8 * 2, 'vae': 8}
    server_target_qps = 16.9
    sdxl_batcher_time_limit = 5
    use_graphs = False
    vboost_slider = 1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.XE8640_H100_SXM_80GBx4
    gpu_batch_size = {'clip1': 32 * 2, 'clip2': 32 * 2, 'unet': 32 * 2, 'vae': 8}
    offline_expected_qps = 2*4
    use_graphs = False
    vboost_slider = 1



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    gpu_batch_size = {'clip1': 8 * 2, 'clip2': 8 * 2, 'unet': 8 * 2, 'vae': 8}
    vboost_slider = 1
    server_target_qps = 16.25
    sdxl_batcher_time_limit = 5
    use_graphs = False

