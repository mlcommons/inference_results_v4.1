# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class CDI_L40SX8(OfflineGPUBaseConfig):
    system = KnownSystem.CDI_L40Sx8

    gpu_batch_size = {'3d-unet': 1}
    offline_expected_qps = 3.8 * 8
    slice_overlap_patch_kernel_cg_impl = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class CDI_L40SX8_HighAccuracy(CDI_L40SX8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class CDI_L40SX16(OfflineGPUBaseConfig):
    system = KnownSystem.CDI_L40Sx16

    gpu_batch_size = {'3d-unet': 1}
    offline_expected_qps = 65
    slice_overlap_patch_kernel_cg_impl = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class CDI_L40SX16_HighAccuracy(CDI_L40SX16):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GX2560M7_H100_SXM_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.GX2560M7_H100_SXM_80GBx4
    # gpu_batch_size = {'3d-unet': 1}
    gpu_batch_size = {'3d-unet': 8}
    offline_expected_qps = 6.8*4


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class GX2560M7_H100_SXM_80GBX4_HighAccuracy(GX2560M7_H100_SXM_80GBX4):
    pass
