# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class CDI_L40SX8(ServerGPUBaseConfig):
    system = KnownSystem.CDI_L40Sx8

    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = {'resnet50': 64}
    gpu_copy_streams = 9
    gpu_inference_streams = 2
    server_target_qps = 164375
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class CDI_L40SX16(ServerGPUBaseConfig):
    system = KnownSystem.CDI_L40Sx16

    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = {'resnet50': 64}
    gpu_copy_streams = 9
    gpu_inference_streams = 2
    server_target_qps = 164375.0
    use_cuda_thread_per_device = True
    use_graphs = True

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class GX2560M7_H100_SXM_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.GX2560M7_H100_SXM_80GBx4
    use_deque_limit = True
    deque_timeout_usec = 3548
    gpu_batch_size = {'resnet50': 273}
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 301250
    use_cuda_thread_per_device = True
    use_graphs = True
    start_from_device = True

