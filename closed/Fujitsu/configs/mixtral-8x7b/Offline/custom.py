# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GX2560M7_H100_SXM_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.GX2560M7_H100_SXM_80GBx4
    gpu_batch_size = {'mixtral-8x7b': 896}
    use_fp8 = True
    offline_expected_qps = 200
    enable_sort = False
    tensor_parallelism = 1
