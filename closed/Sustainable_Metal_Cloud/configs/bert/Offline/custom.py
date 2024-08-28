# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SMC_H100_SXM_80GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.SMC_H100_SXM_80GBX8

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    gpu_batch_size = {'bert': 1280}
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    offline_expected_qps = 9400 * 8
    use_small_tile_gemm_plugin = False
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SMC_H100_SXM_80GBX8_HighAccuracy(SMC_H100_SXM_80GBX8):
    gpu_batch_size = {'bert': 1024}
    offline_expected_qps = 8200 * 8
    precision = "fp16"
    use_fp8 = True
    use_graphs = False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class SMC_H100_SXM_80GBX8_MaxQ(SMC_H100_SXM_80GBX8):
    offline_expected_qps = 54000
    power_limit = 400


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class SMC_H100_SXM_80GBX8_HighAccuracy_MaxQ(SMC_H100_SXM_80GBX8_HighAccuracy):
    power_limit = 450
    offline_expected_qps = 51000

