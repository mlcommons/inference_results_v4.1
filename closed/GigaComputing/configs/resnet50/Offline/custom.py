# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class GIGABYTE_G593_SD1_H200_SXM_141GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.GIGABYTE_G593_SD1_H200_SXM_141GBx8
    gpu_batch_size = {'resnet50': 2048}
    offline_expected_qps = 90000 * 8
   # start_from_device = True
    numa_config = "0:0-13,112-125&1-2:28-41,140-153&3:14-27,126-139&4:56-69,168-181&5-6:84-97,196-209&7:70-83,182-195"

