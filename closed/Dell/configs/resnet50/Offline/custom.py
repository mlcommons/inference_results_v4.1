from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760_H100_PCIE_80GBX2(OfflineGPUBaseConfig):
    system = KnownSystem.R760_H100_PCIe_80GBx2
    gpu_batch_size: dict = {'resnet50': 2048}
    offline_expected_qps = 115000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760XA_H100NVL_PCIE_94GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.R760xa_H100NVL_PCIe_94GBx4
    gpu_batch_size = {'resnet50': 2048}    
    offline_expected_qps = 360000
    gpu_copy_streams = 2
    gpu_inference_streams = 3
    #start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR8620_L4X1(OfflineGPUBaseConfig):
    system = KnownSystem.XR8620_L4x1
    gpu_batch_size = {'resnet50': 32}
    gpu_inference_streams = 2
    gpu_copy_streams = 2
    offline_expected_qps = 13500
    use_graphs = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_L40Sx4(OfflineGPUBaseConfig):
    system = KnownSystem.R760xa_L40Sx4
    gpu_batch_size = {'resnet50': 32}
    offline_expected_qps = 275000
    gpu_copy_streams = 12
    gpu_inference_streams = 4


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.XE8640_H100_SXM_80GBx4
    gpu_batch_size = {'resnet50': 2048}
    offline_expected_qps = 360000
    gpu_inference_streams = 3
    start_from_device = True



@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H200_SXM_141GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H200_SXM_141GBx8
    gpu_batch_size = {'resnet50': 2048}
    offline_expected_qps = 90000 * 8
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    gpu_batch_size = {'resnet50': 2048}
    offline_expected_qps = 90000 * 8
    start_from_device = True


