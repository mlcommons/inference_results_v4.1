# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SMC_H100_SXM_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.SMC_H100_SXM_80GBX8

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig a
lready and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    gpu_batch_size = {'llama2-70b': 1024}
    server_target_qps = 73.5
    tensor_parallelism = 2
    use_fp8 = True
#    vboost_slider = 1



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SMC_H100_SXM_80GBX8_HighAccuracy(SMC_H100_SXM_80GBX8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class SMC_H100_SXM_80GBX8_MaxQ(SMC_H100_SXM_80GBX8):
    server_target_qps = 13.5 * 4
    power_limit = 450


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class SMC_H100_SXM_80GBX8_HighAccuracy_MaxQ(SMC_H100_SXM_80GBX8_HighAccuracy):
    server_target_qps = 13.5 * 4
    power_limit = 450

