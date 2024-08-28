from . import * 

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.XE8640_H100_SXM_80GBx4
    gpu_batch_size = {'mixtral-8x7b': 896}
    use_fp8 = True
    server_target_qps = 44*4
    enable_sort = False
    tensor_parallelism = 1
    vboost_slider = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H200_SXM_141GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H200_SXM_141GBx8
    gpu_batch_size = {'mixtral-8x7b': 3072}
    use_fp8 = True
    offline_expected_qps = 53 * 8
    enable_sort = False
    tensor_parallelism = 1
    vboost_slider = 1


