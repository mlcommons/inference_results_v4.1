# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GX2560M7_H100_SXM_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.GX2560M7_H100_SXM_80GBx4
    use_small_tile_gemm_plugin = False
    gpu_batch_size = {'bert': 1280}
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    offline_expected_qps = 9400*4
    workspace_size = 7516192768    


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class GX2560M7_H100_SXM_80GBX4_HighAccuracy(GX2560M7_H100_SXM_80GBX4):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = {'bert': 1024}
    offline_expected_qps = 8200*4


