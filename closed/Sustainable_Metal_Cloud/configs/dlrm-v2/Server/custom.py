# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SMC_H100_SXM_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.SMC_H100_SXM_80GBX8

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    embedding_weights_on_gpu_part: float = 1.0
    gpu_batch_size = {'dlrm-v2': 51200}
    numa_config = "0-3:0-31,64-95&4-7:32-63,96-127"
    server_num_issue_query_threads = 8
    server_target_qps = 63750 * 8
#    vboost_slider = 1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SMC_H100_SXM_80GBX8_HighAccuracy(SMC_H100_SXM_80GBX8):
    server_target_qps = 340000
    interaction_op_precision = 'fp16'


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class SMC_H100_SXM_80GBX8_MaxQ(SMC_H100_SXM_80GBX8):
    power_limit = 450
    server_target_qps = 50000 * 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class SMC_H100_SXM_80GBX8_HighAccuracy_MaxQ(SMC_H100_SXM_80GBX8_HighAccuracy):
    power_limit = 450
    server_target_qps = 32000 * 8

