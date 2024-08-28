# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class CDI_L40SX8(OfflineGPUBaseConfig):
    system = KnownSystem.CDI_L40Sx8

    embedding_weights_on_gpu_part: float = 1.0
    gpu_batch_size = {'dlrm-v2': 7500}
    offline_expected_qps = 109103

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class CDI_L40SX8_HighAccuracy(CDI_L40SX8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class CDI_L40SX16(OfflineGPUBaseConfig):
    system = KnownSystem.CDI_L40Sx16

    embedding_weights_on_gpu_part: float = 1.0
    gpu_batch_size = {'dlrm-v2': 7500}
    offline_expected_qps = 109103

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class CDI_L40SX16_HighAccuracy(CDI_L40SX16):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GX2560M7_H100_SXM_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.GX2560M7_H100_SXM_80GBx4
    gpu_batch_size = {'dlrm-v2': 600_000}
    embedding_weights_on_gpu_part: float = 1.0
    offline_expected_qps = 80000*4
    numa_config = "0,1:0-47,96-143&2,3:48-95,144-191"
    #numa_config = "0,1:0-47&2,3:48-95"
    #start_from_device = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class GX2560M7_H100_SXM_80GBX4_HighAccuracy(GX2560M7_H100_SXM_80GBX4):
    #offline_expected_qps = 44000*4
    offline_expected_qps = 49000*4
    interaction_op_precision = 'fp16'
