# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class CDI_L40SX8(OfflineGPUBaseConfig):
    system = KnownSystem.CDI_L40Sx8

    gpu_batch_size = {'retinanet': 8}
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    #offline_expected_qps = 750 * 8
    offline_expected_qps = 6500
    run_infer_on_copy_streams = False
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class CDI_L40SX16(OfflineGPUBaseConfig):
    system = KnownSystem.CDI_L40Sx16

    gpu_batch_size = {'retinanet': 8}
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    offline_expected_qps = 6500 * 2
    run_infer_on_copy_streams = False
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class GX2560M7_H100_SXM_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.GX2560M7_H100_SXM_80GBx4
    gpu_batch_size = {'retinanet': 48}
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 1700*4
    run_infer_on_copy_streams = False
    workspace_size = 60000000000

