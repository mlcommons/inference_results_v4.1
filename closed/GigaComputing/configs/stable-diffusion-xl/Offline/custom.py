# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GIGABYTE_G593_SD1_H200_SXM_141GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.GIGABYTE_G593_SD1_H200_SXM_141GBx8
    gpu_batch_size = {'clip1': 32 * 2, 'clip2': 32 * 2, 'unet': 32 * 2, 'vae': 8}
    offline_expected_qps = 2.1 * 8
    use_graphs = False
    vboost_slider = 1


