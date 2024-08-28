# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GIGABYTE_G593_SD1_H200_SXM_141GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.GIGABYTE_G593_SD1_H200_SXM_141GBx8
    offline_expected_qps = 7 * 8 # 6.8 * 8
    gpu_inference_streams = 1
    gpu_copy_streams = 1
    gpu_batch_size = {'3d-unet': 8}
    slice_overlap_patch_kernel_cg_impl = False



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class GIGABYTE_G593_SD1_H200_SXM_141GBX8_HighAccuracy(GIGABYTE_G593_SD1_H200_SXM_141GBX8):
    pass


