from . import *



@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760_H100_PCIE_80GBX2(ServerGPUBaseConfig):
    system = KnownSystem.R760_H100_PCIe_80GBx2
    gpu_batch_size: dict = {'resnet50': 128 }
    deque_timeout_usec: int = 2000
    gpu_copy_streams: int = 4
    gpu_inference_streams: int = 2
    server_target_qps: int = 103000
    use_batcher_thread_per_device: bool = True
    use_cuda_thread_per_device: bool = True
    use_deque_limit: bool = True
    use_graphs: bool = True

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_H100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.R760xa_H100_PCIe_80GBx4
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = {'resnet50': 128}
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 206500
    use_cuda_thread_per_device = True
    use_batcher_thread_per_device = True
    use_graphs = True

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760XA_H100NVL_PCIE_94GBX4(ServerGPUBaseConfig):
    system = KnownSystem.R760xa_H100NVL_PCIe_94GBx4
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = {'retinanet': 8}
    #gpu_batch_size = {'retinanet': 16}
    gpu_inference_streams = 2
    server_target_qps = 1200 * 4
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9640_H100_SXM_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.XE9640_H100_SXM_80GBx4
    start_from_device = True
    gpu_copy_streams = 2
    use_deque_limit = True
    deque_timeout_usec = 32008
    gpu_batch_size = {'retinanet': 14}
    gpu_inference_streams = 3
    server_target_qps = 6737
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_L40Sx4(ServerGPUBaseConfig):
    system = KnownSystem.R760xa_L40Sx4
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 36911
    gpu_batch_size = {'retinanet': 8}
    gpu_inference_streams = 1
    server_target_qps = 3150
    workspace_size = 70000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.XE8640_H100_SXM_80GBx4
    start_from_device = True
    gpu_copy_streams = 2
    use_deque_limit = True
    deque_timeout_usec = 30000
    # gpu_batch_size = {'retinanet': 8}
    gpu_batch_size = {'retinanet': 13}
    gpu_inference_streams = 2
    #server_target_qps = 6710
    #server_target_qps = 6750 - works
    server_target_qps = 6785 # current best
    workspace_size = 60000000000



@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H200_SXM_141GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680_H200_SXM_141GBx8
    start_from_device = True
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = {'retinanet': 8}
    gpu_inference_streams = 2
   server_target_qps = 1700 * 8
    workspace_size = 60000000000



@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    start_from_device = True
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = {'retinanet': 8}
    gpu_inference_streams = 2
    server_target_qps = 1610 * 8
    workspace_size = 60000000000

