from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760XA_H100NVL_PCIE_94GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.R760xa_H100NVL_PCIe_94GBx4
    gpu_batch_size = {'3d-unet': 4}
    offline_expected_qps = 24

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R760XA_H100NVL_PCIE_94GBX4_HighAccuracy(R760XA_H100NVL_PCIE_94GBX4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760_H100_PCIE_80GBX2(OfflineGPUBaseConfig):
    system = KnownSystem.R760_H100_PCIe_80GBx2
    gpu_batch_size: dict = {'3d-unet': 8}
    offline_expected_qps: float = 10.2
    slice_overlap_patch_kernel_cg_impl: bool = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R760_H100_PCIE_80GBX2_HighAccuracy(R760_H100_PCIE_80GBX2):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR8620_L4X1(OfflineGPUBaseConfig):
    system = KnownSystem.XR8620_L4x1
    gpu_batch_size = {'3d-unet': 1}
    offline_expected_qps = 1.2
    slice_overlap_patch_kernel_cg_impl = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XR8620_L4X1_HighAccuracy(XR8620_L4X1):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.XE8640_H100_SXM_80GBx4
    gpu_batch_size = {'3d-unet': 8}
    offline_expected_qps = 6.8*4


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H200_SXM_141GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H200_SXM_141GBx8
    workspace_size = 128000000000
    gpu_batch_size = {'3d-unet': 8}
    offline_expected_qps = 6.8 * 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_H200_SXM_141GBX8_HighAccuracy(XE9680_H200_SXM_141GBX8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
   gpu_batch_size = {'3d-unet': 8}
    offline_expected_qps = 6.8 * 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8_HighAccuracy(XE9680_H100_SXM_80GBX8):
    pass

