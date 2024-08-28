# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class CDI_L40SX8(ServerGPUBaseConfig):
    system = KnownSystem.CDI_L40Sx8

    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = {'retinanet': 8}
    gpu_inference_streams = 2
    server_target_qps = 6200
    workspace_size = 70000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class CDI_L40SX16(ServerGPUBaseConfig):
    system = KnownSystem.CDI_L40Sx16

    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = {'retinanet': 8}
    gpu_inference_streams = 2
    server_target_qps = 11947.05
    workspace_size = 70000000000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class GX2560M7_H100_SXM_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.GX2560M7_H100_SXM_80GBx4
    start_from_device = True
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    # gpu_batch_size = {'retinanet': 8}
    gpu_batch_size = {'retinanet': 16}
    gpu_inference_streams = 2
    server_target_qps = 6800
    workspace_size = 60000000000

