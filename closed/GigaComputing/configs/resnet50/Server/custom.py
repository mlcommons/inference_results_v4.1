# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class GIGABYTE_G593_SD1_H200_SXM_141GBX8(ServerGPUBaseConfig):
    system = KnownSystem.GIGABYTE_G593_SD1_H200_SXM_141GBx8
    gpu_batch_size = {'resnet50': 391}
    use_deque_limit = True
    use_cuda_thread_per_device = True
    use_graphs = True
    start_from_device = True
    server_target_qps = 681050 # 680000 - 680627
    gpu_inference_streams = 1
    gpu_copy_streams = 5
    deque_timeout_usec = 4182
    numa_config = "0:0-13,112-125&1-2:28-41,140-153&3:14-27,126-139&4:56-69,168-181&5-6:84-97,196-209&7:70-83,182-195"

