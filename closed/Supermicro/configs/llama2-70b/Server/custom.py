from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class AS_8125GS_TNHR_H100_SXM_80GBX8(H100_SXM_80GB_TP2x4):
    system = KnownSystem.AS_8125GS_TNHR_H100_SXM_80GBx8
    server_target_qps = 18.34 * 4

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class AS_8125GS_TNHR_H100_SXM_80GBX8_HighAccuracy(H100_SXM_80GB_TP2x4_HighAccuracy):
    system = KnownSystem.AS_8125GS_TNHR_H100_SXM_80GBx8
    server_target_qps = 18.92 * 4

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS_821GE_TNHR_H100_SXM_80GBX8(H100_SXM_80GB_TP2x4):
    system = KnownSystem.SYS_821GE_TNHR_H100_SXM_80GBx8
    server_target_qps = 20.49 * 4

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SYS_821GE_TNHR_H100_SXM_80GBX8_HighAccuracy(H100_SXM_80GB_TP2x4_HighAccuracy):
    system = KnownSystem.SYS_821GE_TNHR_H100_SXM_80GBx8
    server_target_qps = 19.10 * 4

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class AS_4125GS_TNHR2_LCC_H100_SXM_80GBX8(H100_SXM_80GB_TP2x4):
    system = KnownSystem.AS_4125GS_TNHR2_LCC_H100_SXM_80GBx8
    server_target_qps = 19.08 * 4

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class AS_4125GS_TNHR2_LCC_H100_SXM_80GBX8_HighAccuracy(H100_SXM_80GB_TP2x4_HighAccuracy):
    system = KnownSystem.AS_4125GS_TNHR2_LCC_H100_SXM_80GBx8
    server_target_qps = 20.56 * 4

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS_421GE_TNHR2_LCC_H100_SXM_80GBX8(H100_SXM_80GB_TP2x4):
    system = KnownSystem.SYS_421GE_TNHR2_LCC_H100_SXM_80GBx8
    server_target_qps = 19.02 * 4

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SYS_421GE_TNHR2_LCC_H100_SXM_80GBX8_HighAccuracy(H100_SXM_80GB_TP2x4_HighAccuracy):
    system = KnownSystem.SYS_421GE_TNHR2_LCC_H100_SXM_80GBx8
    server_target_qps = 19.02 * 4