from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class AS_8125GS_TNHR_H100_SXM_80GBX8(H100_SXM_80GBx8):
    system = KnownSystem.AS_8125GS_TNHR_H100_SXM_80GBx8
    numa_config = "0,1,2,3:0-47,96-143&4,5,6,7:48-95,144-191"

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class AS_8125GS_TNHR_H100_SXM_80GBX8_HighAccuracy(H100_SXM_80GBx8_HighAccuracy):
    system = KnownSystem.AS_8125GS_TNHR_H100_SXM_80GBx8
    numa_config = "0,1,2,3:0-47,96-143&4,5,6,7:48-95,144-191"
    server_target_qps = 353960
    vboost_slider = 1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS_821GE_TNHR_H100_SXM_80GBX8(H100_SXM_80GBx8):
    system = KnownSystem.SYS_821GE_TNHR_H100_SXM_80GBx8
    numa_config = "0,1,2,3:0-47,96-143&4,5,6,7:48-95,144-191"
    server_target_qps = 68590 * 8

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SYS_821GE_TNHR_H100_SXM_80GBX8_HighAccuracy(H100_SXM_80GBx8_HighAccuracy):
    system = KnownSystem.SYS_821GE_TNHR_H100_SXM_80GBx8
    numa_config = "0,1,2,3:0-47,96-143&4,5,6,7:48-95,144-191"
    server_target_qps = 356360
    vboost_slider = 1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class AS_4125GS_TNHR2_LCC_H100_SXM_80GBX8(H100_SXM_80GBx8):
    system = KnownSystem.AS_4125GS_TNHR2_LCC_H100_SXM_80GBx8
    numa_config = "0,1,2,3:0-95&4,5,6,7:96-191"

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class AS_4125GS_TNHR2_LCC_H100_SXM_80GBX8_HighAccuracy(H100_SXM_80GBx8_HighAccuracy):
    system = KnownSystem.AS_4125GS_TNHR2_LCC_H100_SXM_80GBx8
    numa_config = "0,1,2,3:0-95&4,5,6,7:96-191"
    server_target_qps = 353960
    vboost_slider = 1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS_421GE_TNHR2_LCC_H100_SXM_80GBX8(H100_SXM_80GBx8):
    system = KnownSystem.SYS_421GE_TNHR2_LCC_H100_SXM_80GBx8
    numa_config = "0,1:0-27,112-139&2,3:28-55,140-167&4,5:56-83,168-195&6,7:84-111,196-223"
    server_target_qps = 69490 * 8

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SYS_421GE_TNHR2_LCC_H100_SXM_80GBX8_HighAccuracy(H100_SXM_80GBx8_HighAccuracy):
    system = KnownSystem.SYS_421GE_TNHR2_LCC_H100_SXM_80GBx8
    numa_config = "0,1:0-27,112-139&2,3:28-55,140-167&4,5:56-83,168-195&6,7:84-111,196-223"
    server_target_qps = 357920
    vboost_slider = 1