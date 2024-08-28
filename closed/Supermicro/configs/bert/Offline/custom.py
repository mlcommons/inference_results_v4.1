from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class AS_8125GS_TNHR_H100_SXM_80GBX8(H100_SXM_80GBx8):
    system = KnownSystem.AS_8125GS_TNHR_H100_SXM_80GBx8
    start_from_device = True
    vboost_slider = 1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class AS_8125GS_TNHR_H100_SXM_80GBX8_HighAccuracy(H100_SXM_80GBx8_HighAccuracy):
    system = KnownSystem.AS_8125GS_TNHR_H100_SXM_80GBx8
    start_from_device = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS_821GE_TNHR_H100_SXM_80GBX8(H100_SXM_80GBx8):
    system = KnownSystem.SYS_821GE_TNHR_H100_SXM_80GBx8
    start_from_device = True
    vboost_slider = 1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SYS_821GE_TNHR_H100_SXM_80GBX8_HighAccuracy(H100_SXM_80GBx8_HighAccuracy):
    system = KnownSystem.SYS_821GE_TNHR_H100_SXM_80GBx8
    start_from_device = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class AS_4125GS_TNHR2_LCC_H100_SXM_80GBX8(H100_SXM_80GBx8):
    system = KnownSystem.AS_4125GS_TNHR2_LCC_H100_SXM_80GBx8
    start_from_device = True
    vboost_slider = 1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class AS_4125GS_TNHR2_LCC_H100_SXM_80GBX8_HighAccuracy(H100_SXM_80GBx8_HighAccuracy):
    system = KnownSystem.AS_4125GS_TNHR2_LCC_H100_SXM_80GBx8
    start_from_device = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS_421GE_TNHR2_LCC_H100_SXM_80GBX8(H100_SXM_80GBx8):
    system = KnownSystem.SYS_421GE_TNHR2_LCC_H100_SXM_80GBx8
    start_from_device = True
    vboost_slider = 1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SYS_421GE_TNHR2_LCC_H100_SXM_80GBX8_HighAccuracy(H100_SXM_80GBx8_HighAccuracy):
    system = KnownSystem.SYS_421GE_TNHR2_LCC_H100_SXM_80GBx8
    start_from_device = True