# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ESC8000A_E12_H100X8(H100_PCIe_80GBx8):
    system = KnownSystem.ESC8000A_E12_H100x8

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ESC8000A_E12_H100X8_HighAccuracy(ESC8000A_E12_H100X8):
    pass




@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ESC_N8_E11_H100x8(H100_SXM_80GBx8):
    system = KnownSystem.ESC_N8_E11_H100x8
    gpu_batch_size : {'gptj' : 256}


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ESC_N8_E11_H100X8_HighAccuracy(ESC_N8_E11_H100x8):
    system = KnownSystem.ESC_N8_E11_H100x8
    gpu_batch_size : {'gptj' : 256}



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ESC4000A_E12_4XH100(ServerGPUBaseConfig):
    system = KnownSystem.ESC4000A_E12_4XH100
    gpu_batch_size = {'gptj': 512}
    use_fp8 = True
    server_target_qps = 90

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ESC4000A_E12_4XH100_HighAccuracy(ESC4000A_E12_4XH100):
    server_target_qps = 88
