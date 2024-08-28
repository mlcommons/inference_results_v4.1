# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class CDI_L40SX8(ServerGPUBaseConfig):
    system = KnownSystem.CDI_L40Sx8
    gpu_batch_size = {"llama2-70b": 1024}
    use_fp8 = True
    tensor_parallelism = 4
    server_target_qps = 11

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class CDI_L40SX8_HighAccuracy(CDI_L40SX8):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GX2560M7_H100_SXM_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.GX2560M7_H100_SXM_80GBx4
    gpu_batch_size = {'llama2-70b': 1024}
    use_fp8 = True
    tensor_parallelism = 2
    #server_target_qps = 36.5
    server_target_qps = 33.8
    vboost_slider = 1
    enable_sort = False

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class GX2560M7_H100_SXM_80GBX4_HighAccuracy(GX2560M7_H100_SXM_80GBX4):
    gpu_batch_size = {'llama2-70b': 1024}
    use_fp8 = True
    tensor_parallelism = 2
    server_target_qps = 33
    vboost_slider = 1
    enable_sort = False


