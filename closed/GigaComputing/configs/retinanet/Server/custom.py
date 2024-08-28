# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class GIGABYTE_G593_SD1_H200_SXM_141GBX8(ServerGPUBaseConfig):
    system = KnownSystem.GIGABYTE_G593_SD1_H200_SXM_141GBx8

    gpu_copy_streams = 4
    gpu_batch_size = {'retinanet': 12}
    server_target_qps = 1751 * 8
    gpu_inference_streams = 2
    workspace_size = 60000000000
