from . import * 

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760_H100_PCIE_80GBX2(OfflineGPUBaseConfig):
    system = KnownSystem.R760_H100_PCIe_80GBx2
    gpu_batch_size: dict = {'gptj': 192}
    use_fp8 = True
    offline_expected_qps = 50 
    enable_sort = False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R760_H100_PCIE_80GBX2_HighAccuracy(R760_H100_PCIE_80GBX2):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.XE8640_H100_SXM_80GBx4
    gpu_batch_size = {'gptj': 192}
    use_fp8 = True
    offline_expected_qps = 36*4
    enable_sort = False

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4_HighAccuracy(XE8640_H100_SXM_80GBx4):
    pass



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H200_SXM_141GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H200_SXM_141GBx8
    gpu_batch_size = {'gptj': 396}
    use_fp8 = True
    offline_expected_qps = 36 * 8

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_H200_SXM_141GBX8_HighAccuracy(XE9680_H200_SXM_141GBX8):
    pass



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    use_fp8 = True
    gpu_batch_size = {'gptj': 192}
    offline_expected_qps = 36 * 8
    enable_sort = False

