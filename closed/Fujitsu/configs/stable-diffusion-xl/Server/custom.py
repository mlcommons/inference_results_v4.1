# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GX2560M7_H100_SXM_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.GX2560M7_H100_SXM_80GBx4
    gpu_batch_size = {'clip1': 8 * 2, 'clip2': 8 * 2, 'unet': 8 * 2, 'vae': 8}
    server_target_qps = 2*4
    sdxl_batcher_time_limit = 5  # or 3 or 4
    use_graphs = False  # disable to meet latency constraint for x1
    
