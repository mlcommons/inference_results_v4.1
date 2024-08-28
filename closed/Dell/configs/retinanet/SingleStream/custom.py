from . import *

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR8620_L4X1(MultiStreamGPUBaseConfig):
    system = KnownSystem.XR8620_L4x1
    multi_stream_expected_latency_ns = 40000000

