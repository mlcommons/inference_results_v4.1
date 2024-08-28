from . import * 

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR8620_L4X1(OfflineGPUBaseConfig):
    system = KnownSystem.XR8620_L4x1
    gpu_batch_size = {'retinanet': 2}
    gpu_copy_streams = 3
    gpu_inference_streams = 2
    offline_expected_qps = 255 #170
    run_infer_on_copy_streams = False
    workspace_size = 20000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760XA_H100NVL_PCIE_94GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.R760xa_H100NVL_PCIe_94GBx4
    gpu_batch_size = {'retinanet': 48}
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 1700 * 4
    run_infer_on_copy_streams = False
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.XE8640_H100_SXM_80GBx4
    gpu_batch_size = {'retinanet': 30}
    gpu_copy_streams = 3
    gpu_inference_streams = 2
    offline_expected_qps = 6800
    run_infer_on_copy_streams = False
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H200_SXM_141GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H200_SXM_141GBx8
    gpu_batch_size = {'retinanet': 48}
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 14605
    run_infer_on_copy_streams = False
    workspace_size = 128000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    gpu_batch_size = {'retinanet': 48}
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    #offline_expected_qps = 1700 * 8
    offline_expected_qps = 14486
    run_infer_on_copy_streams = False
    workspace_size = 60000000000

