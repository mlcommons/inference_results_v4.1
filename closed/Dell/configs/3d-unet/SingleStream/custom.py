from . import * 

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR8620_L4X1(SingleStreamGPUBaseConfig):
    system = KnownSystem.XR8620_L4x1
    gpu_batch_size = {'3d-unet': 4}
    single_stream_expected_latency_ns = 572434000
    slice_overlap_patch_kernel_cg_impl = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XR8620_L4X1_HighAccuracy(XR8620_L4X1):
    pass


