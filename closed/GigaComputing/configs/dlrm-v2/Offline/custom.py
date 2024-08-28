# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GIGABYTE_G593_SD1_H200_SXM_141GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.GIGABYTE_G593_SD1_H200_SXM_141GBx8
    gpu_batch_size = {'dlrm-v2': 600_000}
    embedding_weights_on_gpu_part: float = 1.0
    offline_expected_qps = 90000 * 8  # 635000
    start_from_device = True
    vboost_slider = 1
    # numa_config = "0-3:0-55,112-167&4-7:56-111,168-223"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class GIGABYTE_G593_SD1_H200_SXM_141GBX8_HighAccuracy(GIGABYTE_G593_SD1_H200_SXM_141GBX8):
    offline_expected_qps = 90000 * 8 # 390000
    interaction_op_precision = 'fp16'


