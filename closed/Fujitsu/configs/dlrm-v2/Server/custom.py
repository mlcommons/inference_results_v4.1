# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class CDI_L40SX8(ServerGPUBaseConfig):
    system = KnownSystem.CDI_L40Sx8

    embedding_weights_on_gpu_part = 1.0
    gpu_batch_size = {'dlrm-v2': 7500}
    server_target_qps = 180000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class CDI_L40SX8_HighAccuracy(CDI_L40SX8):
    server_target_qps = 85000
    interaction_op_precision = 'fp16'
    top_mlp_precision = 'fp16'
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class CDI_L40SX16(ServerGPUBaseConfig):
    system = KnownSystem.CDI_L40Sx16

    embedding_weights_on_gpu_part = 1.0
    gpu_batch_size = {'dlrm-v2': 7500}
    #server_target_qps = 91796
    server_target_qps = 90000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class CDI_L40SX16_HighAccuracy(CDI_L40SX16):
    #server_target_qps = 94296.875
    server_target_qps = 90000
    interaction_op_precision = 'fp16'
    top_mlp_precision = 'fp16'
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GX2560M7_H100_SXM_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.GX2560M7_H100_SXM_80GBx4
    gpu_batch_size = {'dlrm-v2': 51200 * 2}
    embedding_weights_on_gpu_part: float = 1.0
    #server_target_qps = 295250.0
    server_target_qps = 293250.0
    server_num_issue_query_threads = 4
    #numa_config = "0,1:0-47&2,3:48-95"
    numa_config = "0,1:0-47,96-143&2,3:48-95,144-191"

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class GX2560M7_H100_SXM_80GBX4_HighAccuracy(GX2560M7_H100_SXM_80GBX4):
    server_target_qps = 179000.0
    interaction_op_precision = 'fp16'
