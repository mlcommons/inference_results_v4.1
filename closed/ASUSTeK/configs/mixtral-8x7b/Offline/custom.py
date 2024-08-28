# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ESC8000A_E12_H100X8(OfflineGPUBaseConfig):
    system = KnownSystem.ESC8000A_E12_H100x8

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):

    gpu_batch_size = {'mixtral-8x7b': 1024}
    use_fp8 = True
    offline_expected_qps = 250
    enable_sort = False
    #tensor_parallelism = 1
    max_num_tokens = 8192
    #vboost_slider = 1


