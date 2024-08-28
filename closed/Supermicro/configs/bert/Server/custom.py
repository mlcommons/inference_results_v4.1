from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class AS_8125GS_TNHR_H100_SXM_80GBX8(H100_SXM_80GBx8):
    system = KnownSystem.AS_8125GS_TNHR_H100_SXM_80GBx8
    start_from_device = True
    gpu_batch_size = {'bert': 171}
    server_target_qps = 57480
    vboost_slider = 1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class AS_8125GS_TNHR_H100_SXM_80GBX8_HighAccuracy(H100_SXM_80GBx8_HighAccuracy):
    system = KnownSystem.AS_8125GS_TNHR_H100_SXM_80GBx8
    start_from_device = True
    gpu_batch_size = {'bert': 292}
    gpu_copy_streams = 6
    server_target_qps = 50720

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS_821GE_TNHR_H100_SXM_80GBX8(H100_SXM_80GBx8):
    system = KnownSystem.SYS_821GE_TNHR_H100_SXM_80GBx8
    start_from_device = True
    gpu_batch_size = {'bert': 171}
    server_target_qps = 57920
    vboost_slider = 1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SYS_821GE_TNHR_H100_SXM_80GBX8_HighAccuracy(H100_SXM_80GBx8_HighAccuracy):
    system = KnownSystem.SYS_821GE_TNHR_H100_SXM_80GBx8
    start_from_device = True
    gpu_batch_size = {'bert': 292}
    gpu_copy_streams = 6
    server_target_qps = 51560

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class AS_4125GS_TNHR2_LCC_H100_SXM_80GBX8(H100_SXM_80GBx8):
    system = KnownSystem.AS_4125GS_TNHR2_LCC_H100_SXM_80GBx8
    start_from_device = True
    gpu_batch_size = {'bert': 171}
    server_target_qps = 57840
    vboost_slider = 1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class AS_4125GS_TNHR2_LCC_H100_SXM_80GBX8_HighAccuracy(H100_SXM_80GBx8_HighAccuracy):
    system = KnownSystem.AS_4125GS_TNHR2_LCC_H100_SXM_80GBx8
    start_from_device = True
    gpu_batch_size = {'bert': 292}
    gpu_copy_streams = 6
    server_target_qps = 51040

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS_421GE_TNHR2_LCC_H100_SXM_80GBX8(H100_SXM_80GBx8):
    system = KnownSystem.SYS_421GE_TNHR2_LCC_H100_SXM_80GBx8
    start_from_device = True
    gpu_batch_size = {'bert': 171}
    server_target_qps = 58920
    vboost_slider = 1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SYS_421GE_TNHR2_LCC_H100_SXM_80GBX8_HighAccuracy(H100_SXM_80GBx8_HighAccuracy):
    system = KnownSystem.SYS_421GE_TNHR2_LCC_H100_SXM_80GBx8
    start_from_device = True
    gpu_batch_size = {'bert': 292}
    gpu_copy_streams = 6
    server_target_qps = 51960