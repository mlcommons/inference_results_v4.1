# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SMC_H100_SXM_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.SMC_H100_SXM_80GBX8

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    gpu_batch_size = {'clip1': 8 * 2, 'clip2': 8 * 2, 'unet': 8 * 2, 'vae': 8}
    sdxl_batcher_time_limit = 5
    server_target_qps = 16
    use_graphs = False
#    vboost_slider = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class SMC_H100_SXM_80GBX8_MaxQ(SMC_H100_SXM_80GBX8):
    server_target_qps = 11
    power_limit = 350

