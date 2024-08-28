# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class SMC_H100_SXM_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.SMC_H100_SXM_80GBX8

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    deque_timeout_usec = 30000
    gpu_inference_streams = 2
    gpu_copy_streams = 4
    gpu_batch_size = {'retinanet': 8}
    server_target_qps = 1610 * 8
#    start_from_device = True
    use_deque_limit = True
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class SMC_H100_SXM_80GBX8_MaxQ(SMC_H100_SXM_80GBX8):
    power_limit = 350
    server_target_qps = 1100 * 8

