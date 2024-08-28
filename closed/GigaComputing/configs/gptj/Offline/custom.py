# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GIGABYTE_G593_SD1_H200_SXM_141GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.GIGABYTE_G593_SD1_H200_SXM_141GBx8
    gpu_batch_size = {'gptj': 396}
    use_fp8 = True
    offline_expected_qps = 36 * 8

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class GIGABYTE_G593_SD1_H200_SXM_141GBX8_HighAccuracy(GIGABYTE_G593_SD1_H200_SXM_141GBX8):
    pass


