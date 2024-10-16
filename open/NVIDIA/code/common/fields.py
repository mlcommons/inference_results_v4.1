# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import re
from argparse import ArgumentParser
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Dict, Tuple, Optional, Union
from typeguard import check_type, TypeCheckError

from code.common.constants import *
from code.common.systems.systems import SystemConfiguration
from code.common.systems.system_list import KnownSystem, SystemClassifications

_LWIS_BENCHMARKS = (Benchmark.ResNet50,
                    Benchmark.Retinanet,
                    Benchmark.UNET3D)
_CV_BENCHMARKS = (Benchmark.ResNet50,
                  Benchmark.Retinanet)
_HARNESS_ACTIONS = (Action.GenerateConfFiles,  # Since .conf files are a dependency of run_harness
                    Action.RunHarness,
                    Action.RunAuditHarness)


@dataclass(frozen=True)
class Field:
    """
    Represents a configuration parameter used for any benchmark workload.
    """

    name: str
    """str: Identifier for this field"""

    description: str
    """str: Description of this field's meaning and usage"""

    value_type: type
    """type: The expected type of the value of the field"""

    default: Any = None
    """Any: The default value of this Field"""

    supported_actions: Tuple[Action, ...] = tuple(Action)
    """Tuple[Action, ...]: Actions this Field is used in"""

    supported_benchmarks: Tuple[Benchmark, ...] = tuple(Benchmark)
    """Tuple[Benchmark, ...]: Benchmarks this Field is used in"""

    supported_scenarios: Tuple[Scenario, ...] = tuple(Scenario)
    """Tuple[Scenario, ...]: Scenarios this Field is used in"""

    supported_systems: Tuple[Callable[KnownSystem, bool], ...] = (lambda s: True,)
    """Tuple[System, ...]: Systems this Field is used in"""

    supported_harnesses: Tuple[HarnessType, ...] = tuple(HarnessType)
    """Tuple[HarnessType, ...]: Harnesses this Field can be used in"""

    supported_power_settings: Tuple[PowerSetting, ...] = tuple(PowerSetting)
    """Tuple[HarnessType, ...]: Harnesses this Field can be used in"""

    required: bool = False
    """bool: Whether or not this Field is required to run the workload"""

    no_argparse: bool = False
    """bool: If True, this Field should not be added to argparsers"""

    argparse_opts: Optional[Dict[str, Any]] = None
    """Dict[str, Any]: Optional kwargs to use for Argparse.Argument"""

    def add_to_argparser(self, argparser: ArgumentParser, allow_argparse_default=False):
        """
        Adds this field to an argparser as a command line argument to be parsed. If this Field has a value_type of bool,
        it is treated as a store_true argument (i.e. --verbose).

        If no_argparse is True, this method returns immediately without adding to the argparser.

        Args:
            argparser (argparse.ArgumentParser):
                The argparser to add this field to as an argument.
            allow_argparse_default (bool):
                If False, disallows using 'default' in argparse_opts. This is useful to prevent bugs where a default
                argparse value will override any changes made in the BenchmarkConfiguration, as CLI overrides take the
                highest precedence for runtime arguments. Default: False.
        """
        if self.no_argparse:
            return

        kwargs = dict() if self.argparse_opts is None else self.argparse_opts
        if not allow_argparse_default and "default" in kwargs:
            raise RuntimeError(" ".join([
                f"Field({self.name}) contains 'default' in argparse_opts, which is disallowed.",
                "If you wish to specify a default value, use the Field.default instead."
            ]))

        if self.value_type is bool:
            argparser.add_argument(f"--{self.name}", help=self.description, action="store_true", **kwargs)
        # Parse batch size inputs as strings to construct component - component batch size pairs
        elif self.name in ("gpu_batch_size", "dla_batch_size"):
            argparser.add_argument(f"--{self.name}", help=self.description, type=str, **kwargs)
        else:
            argparser.add_argument(f"--{self.name}", help=self.description, type=self.value_type, **kwargs)

    def supports(self, action: Optional[Action], benchmark: Benchmark, scenario: Scenario,
                 system: KnownSystem, workload_setting: WorkloadSetting) -> bool:
        """
        Whether or not this Field supports the given workload.

        Args:
            action (Optional[Action]):
                The Action of the workload. If None, unused during the check.
            benchmark (Benchmark):
                The Benchmark of the workload
            scenario (Scenario):
                The Scenario of the workload
            system (KnownSystem):
                The System running the workload
            workload_setting (WorkloadSetting):
                The WorkloadSetting field of the config to get fields for

        Returns:
            bool: True if all of the parameters of the workload are supported by this Field. False otherwise.
            harness_type is only checked if `action` is Action.RunHarness, Action.RunAuditHarness, or None.
        """
        return (action is None or action in self.supported_actions) and \
            benchmark in self.supported_benchmarks and \
            scenario in self.supported_scenarios and \
            all(f(system) for f in self.supported_systems) and \
            (action not in (list(_HARNESS_ACTIONS) + [None]) or
                workload_setting.harness_type in self.supported_harnesses) and \
            workload_setting.power_setting in self.supported_power_settings


@unique
class Fields(Enum):
    """
    Defines a set of known Fields for BenchmarkConfigurations.
    """

    gpu_batch_size: Field = Field(
        "gpu_batch_size",
        "Batch size to use for the GPU.",
        supported_systems=(SystemClassifications.gpu_based,),
        value_type=Dict[Union[str, AliasedNameEnum], int],
        required=True)

    dla_batch_size: Field = Field(
        "dla_batch_size",
        "Batch size to use for the DLA. NVIDIA SoC-only argument.",
        value_type=Dict[Union[str, AliasedNameEnum], int],
        supported_benchmarks=_CV_BENCHMARKS,
        supported_scenarios=(Scenario.Offline,),
        supported_systems=(SystemClassifications.is_soc,),
        required=False)

    verbose: Field = Field(
        "verbose",
        "Whether to use verbose output.",
        value_type=bool)

    verbose_nvtx: Field = Field(
        "verbose_nvtx",
        "Turn ProfilingVerbosity to kDETAILED so NVTX prints layer detail. Used only when profiling.",
        value_type=bool)

    verbose_glog: Field = Field(
        "verbose_glog",
        "Verbosity level for glogger.",
        value_type=int)

    workspace_size: Field = Field(
        "workspace_size",
        "The maximum size (in bytes) of temporary workspace that any layer in the network can use in TRT engine builder.",
        value_type=int,
        supported_actions=(Action.GenerateEngines,))

    power_limit: Field = Field(
        "power_limit",
        "Set power upper limit to the specified value.",
        value_type=int,
        supported_power_settings=(PowerSetting.MaxQ,))

    cpu_freq: Field = Field(
        "cpu_freq",
        "Set cpu frequency to the specified value.",
        value_type=int,
        supported_power_settings=(PowerSetting.MaxQ,))

    soc_gpu_freq: Field = Field(
        "soc_gpu_freq",
        "Set power upper limit to the specified value.",
        value_type=int,
        supported_power_settings=(PowerSetting.MaxQ,),
        supported_systems=(SystemClassifications.is_soc,))

    soc_dla_freq: Field = Field(
        "soc_dla_freq",
        "Set dla clock to the specified value.",
        value_type=int,
        supported_power_settings=(PowerSetting.MaxQ,),
        supported_systems=(SystemClassifications.is_soc,))

    soc_cpu_freq: Field = Field(
        "soc_cpu_freq",
        "Set cpu clock to the specified value.",
        value_type=int,
        supported_power_settings=(PowerSetting.MaxQ,),
        supported_systems=(SystemClassifications.is_soc,))

    soc_emc_freq: Field = Field(
        "soc_emc_freq",
        "Set emc clock to specified value.",
        value_type=int,
        supported_power_settings=(PowerSetting.MaxQ,),
        supported_systems=(SystemClassifications.is_soc,))

    soc_pva_freq: Field = Field(
        "soc_pva_freq",
        "Set pva clock to specified value.",
        value_type=int,
        supported_power_settings=(PowerSetting.MaxQ,),
        supported_systems=(SystemClassifications.is_soc,))

    orin_num_cores: Field = Field(
        "orin_num_cores",
        "Enable orin_num_cores CPU cores on Orin, turn off others.",
        value_type=int,
        supported_power_settings=(PowerSetting.MaxQ,),
        supported_systems=(SystemClassifications.is_orin,))

    orin_skip_maxq_reset: Field = Field(
        "orin_skip_maxq_reset",
        "Skip resetting board state via nvpmodel -m 0.",
        value_type=bool,
        default=False,
        supported_power_settings=(PowerSetting.MaxQ,),
        supported_systems=(SystemClassifications.is_orin,))

    data_dir: Field = Field(
        "data_dir",
        "Directory containing raw, unprocessed dataset.",
        value_type=str,
        default=os.environ.get("DATA_DIR", "build/data"))

    preprocessed_data_dir: Field = Field(
        "preprocessed_data_dir",
        "Directory containing preprocessed dataset.",
        value_type=str,
        default=os.environ.get("PREPROCESSED_DATA_DIR", "build/preprocessed_data"))

    precision: Field = Field(
        "precision",
        "Precision to use for the network",
        value_type=str,
        required=True,
        argparse_opts={"choices": Precision.as_strings()})

    input_dtype: Field = Field(
        "input_dtype",
        "Precision of the input",
        value_type=str,
        required=True,
        argparse_opts={"choices": Precision.as_strings()})

    input_format: Field = Field(
        "input_format",
        "Format/layout of the input",
        value_type=str,
        required=True,
        supported_systems=(SystemClassifications.gpu_based,),
        argparse_opts={"choices": InputFormats.as_strings()})

    force_calibration: Field = Field(
        "force_calibration",
        "Run quantization calibration, even if the cache exists.",
        value_type=bool,
        supported_actions=(Action.GenerateEngines, Action.Calibrate))

    calib_batch_size: Field = Field(
        "calib_batch_size",
        "Batch size to use when calibrating",
        value_type=int,
        supported_actions=(Action.Calibrate,))

    calib_max_batches: Field = Field(
        "calib_max_batches",
        "Number of batches to run for calibration.",
        value_type=int,
        supported_actions=(Action.Calibrate,))

    cache_file: Field = Field(
        "cache_file",
        "Path to calibration cache.",
        value_type=str,
        supported_actions=(Action.GenerateEngines, Action.Calibrate))

    calib_data_map: Field = Field(
        "calib_data_map",
        "Path to the data map of the calibration set.",
        value_type=str,
        supported_actions=(Action.Calibrate,))

    benchmark: Field = Field(
        "benchmark",
        "Name of the benchmark.",
        value_type=Benchmark,
        argparse_opts={"choices": Benchmark.as_strings()},
        required=True)

    scenario: Field = Field(
        "scenario",
        "Name of the scenario.",
        value_type=Scenario,
        argparse_opts={"choices": Scenario.as_strings()},
        required=True)

    system: Field = Field(
        "system",
        "System object. Note this cannot be passed in via CLI, but is used to validate BenchmarkConfiguration objects.",
        value_type=SystemConfiguration,
        no_argparse=True,
        required=True)

    dla_core: Field = Field(
        "dla_core",
        "DLA Core ID to use.",
        value_type=int,
        supported_actions=(Action.GenerateEngines,),
        supported_benchmarks=_CV_BENCHMARKS,
        supported_scenarios=(Scenario.Offline,),
        supported_systems=(SystemClassifications.is_soc,))

    model_path: Field = Field(
        "model_path",
        "Path to the model weights.",
        value_type=str)

    quant_model_path: Field = Field(
        "quant_model_path",
        "Path to the quantized model weights.",
        value_type=str)

    active_sms: Field = Field(
        "active_sms",
        "Percentage of active SMs while generating engines.",
        value_type=int,
        supported_actions=(Action.GenerateEngines,))

    log_dir: Field = Field(
        "log_dir",
        "Directory for all output logs.",
        value_type=str,
        default=os.environ.get("LOG_DIR", "build/logs/default"))

    use_graphs: Field = Field(
        "use_graphs",
        "Enable CUDA graphs.",
        value_type=bool,
        default=False,
        supported_actions=_HARNESS_ACTIONS + (Action.GenerateEngines,),
        supported_systems=(SystemClassifications.gpu_based,))

    disable_beta1_smallk: Field = Field(
        "disable_beta1_smallk",
        "Disable the beta=1 smallk plugins for resnet/resnext layers",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS + (Action.GenerateEngines,),
        supported_benchmarks=(Benchmark.ResNet50, Benchmark.Retinanet),
        supported_systems=(SystemClassifications.gpu_based,))

    energy_aware_kernels: Field = Field(
        "energy_aware_kernels",
        "Override layers with energy aware kernel selection",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS + (Action.GenerateEngines,),
        supported_benchmarks=(Benchmark.ResNet50, Benchmark.Retinanet, Benchmark.BERT),
        supported_systems=(SystemClassifications.gpu_based,))

    use_small_tile_gemm_plugin: Field = Field(
        "use_small_tile_gemm_plugin",
        "Enable the Small Tile GEMM Plugin for TensorRT",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS + (Action.GenerateEngines,),
        supported_benchmarks=(Benchmark.BERT,),
        supported_systems=(SystemClassifications.gpu_based,))

    gemm_plugin_fairshare_cache_size: Field = Field(
        "gemm_plugin_fairshare_cache_size",
        "Cache size (i.e. number of cache entries) for the FairShare cache in the GEMM Plugin",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS + (Action.GenerateEngines,),
        supported_benchmarks=(Benchmark.BERT,),
        supported_systems=(SystemClassifications.gpu_based,))

    devices: Field = Field(
        "devices",
        "Comma-separated list of numbered devices.",
        value_type=str,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.ResNet50,
                              Benchmark.Retinanet,
                              Benchmark.UNET3D,
                              Benchmark.BERT,
                              Benchmark.GPTJ,
                              Benchmark.LLAMA2,
                              Benchmark.Mixtral8x7B,
                              Benchmark.SDXL))  # For some reason only harness_dlrm does not support this.

    map_path: Field = Field(
        "map_path",
        "Path to map file for samples. Not used if coalesced_tensor is True (BERT, DLRMv2).",
        value_type=str,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.ResNet50,
                              Benchmark.Retinanet,
                              Benchmark.UNET3D),
        required=True)

    tensor_path: Field = Field(
        "tensor_path",
        "Path to preprocessed samples in .npy format",
        value_type=str,
        supported_actions=_HARNESS_ACTIONS,
        required=True)

    performance_sample_count: Field = Field(
        "performance_sample_count",
        "Number of samples to load in performance set. 0=use default",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS)

    performance_sample_count_override: Field = Field(
        "performance_sample_count_override",
        "Number of samples to load in performance set; overriding performance_sample_count. 0=don't override",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS)

    gpu_copy_streams: Field = Field(
        "gpu_copy_streams",
        """Number of copy streams to use for GPU. Used in generate_engines as well to determine number of TensorRT
        optimization profiles""",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS + (Action.GenerateEngines,),
        supported_systems=(SystemClassifications.gpu_based,))

    gpu_inference_streams: Field = Field(
        "gpu_inference_streams",
        """Number of inference streams to use for GPU. Used in generate_engines as well to determine number of TensorRT
        optimization profiles""",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS + (Action.GenerateEngines,),
        supported_systems=(SystemClassifications.gpu_based,))

    dla_copy_streams: Field = Field(
        "dla_copy_streams",
        "Number of copy streams to use for DLA",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=_CV_BENCHMARKS,
        supported_scenarios=(Scenario.Offline,),
        supported_systems=(SystemClassifications.is_soc,))

    dla_inference_streams: Field = Field(
        "dla_inference_streams",
        "Number of inference streams to use for DLA",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=_CV_BENCHMARKS,
        supported_scenarios=(Scenario.Offline,),
        supported_systems=(SystemClassifications.is_soc,))

    run_infer_on_copy_streams: Field = Field(
        "run_infer_on_copy_streams",
        "Run inference on copy streams",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS)

    warmup_duration: Field = Field(
        "warmup_duration",
        "Minimum duration to perform warmup for",
        value_type=float,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=_LWIS_BENCHMARKS + (Benchmark.DLRMv2,))

    use_direct_host_access: Field = Field(
        "use_direct_host_access",
        "Use direct access to host memory for all devices. (SoC Unified Memory)",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=_CV_BENCHMARKS + (Benchmark.UNET3D,),
        supported_systems=(SystemClassifications.is_soc,))

    use_deque_limit: Field = Field(
        "use_deque_limit",
        "Use a max number of elements dequed from work queue (LWIS only)",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=_LWIS_BENCHMARKS)

    deque_timeout_usec: Field = Field(
        "deque_timeout_usec",
        "Timeout in us for deque from work queue (LWIS only)",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=_LWIS_BENCHMARKS + (Benchmark.BERT,))

    use_batcher_thread_per_device: Field = Field(
        "use_batcher_thread_per_device",
        "Enable a separate batcher thread per device",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=_LWIS_BENCHMARKS + (Benchmark.DLRMv2,))

    use_cuda_thread_per_device: Field = Field(
        "use_cuda_thread_per_device",
        "Enable a separate cuda thread per device",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=_LWIS_BENCHMARKS,
        supported_systems=(SystemClassifications.gpu_based,))

    start_from_device: Field = Field(
        "start_from_device",
        "Assuming that inputs start from device memory in QSL",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_systems=(SystemClassifications.start_from_device_enabled,))

    end_on_device: Field = Field(
        "end_on_device",
        "Allows output to remain device memory for QuerySampleComplete. As of v1.1, only supported on 3D-UNET.",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.UNET3D,),
        supported_systems=(SystemClassifications.end_on_device_enabled,))

    max_dlas: Field = Field(
        "max_dlas",
        "Max number of DLAs to use per device",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=_CV_BENCHMARKS,
        supported_systems=(SystemClassifications.is_soc,))

    coalesced_tensor: Field = Field(
        "coalesced_tensor",
        "Turn on if all the samples are coalesced into one single npy file (LWIS Only)",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.BERT, Benchmark.GPTJ, Benchmark.LLAMA2, Benchmark.Mixtral8x7B, Benchmark.DLRMv2))

    assume_contiguous: Field = Field(
        "assume_contiguous",
        "Assume that the data in a query is already contiguous (LWIS Only)",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=tuple(set(_LWIS_BENCHMARKS) - set((Benchmark.UNET3D,))))

    complete_threads: Field = Field(
        "complete_threads",
        "Number of threads per device for sending responses",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=_LWIS_BENCHMARKS + (Benchmark.DLRMv2,))

    use_same_context: Field = Field(
        "use_same_context",
        "Use the same TRT context for all copy streams (shape must be static and gpu_inference_streams must be 1).",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=_LWIS_BENCHMARKS)

    # TODO: From here down, I stopped checking for supported_* fields to save time. Add these after.
    unet3d_sw_gaussian_patch_path: Field = Field(
        "unet3d_sw_gaussian_patch_path",
        "Path to the numpy file holding preconditioned Gaussian patches used in 3D-UNet-KiTS19.",
        value_type=str,
        default=os.environ.get("PREPROCESSED_DATA_DIR", "build/preprocessed_data") + "/KiTS19/etc/gaussian_patches.npy",
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.UNET3D,))

    slice_overlap_patch_kernel_cg_impl: Field = Field(
        "slice_overlap_patch_kernel_cg_impl",
        "Use 3D-UNet patch kernel implemented using cooperative-group; if false, kernel impl using CPU implicit sync is used",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.UNET3D,))

    mlperf_conf_path: Field = Field(
        "mlperf_conf_path",
        "Path to mlperf.conf",
        value_type=str,
        supported_actions=_HARNESS_ACTIONS)

    user_conf_path: Field = Field(
        "user_conf_path",
        "Path to user.conf",
        value_type=str,
        supported_actions=_HARNESS_ACTIONS)

    test_mode: Field = Field(
        "test_mode",
        "Testing mode for Loadgen",
        value_type=str,
        supported_actions=_HARNESS_ACTIONS,
        argparse_opts={"choices": ["SubmissionRun", "AccuracyOnly", "PerformanceOnly", "FindPeakPerformance"]})

    model_opt: Field = Field(
        "model_opt",
        "Model optimization method",
        value_type=str,
        supported_actions=_HARNESS_ACTIONS,
        argparse_opts={"choices": ["NoOpt", "HeadPruned", "DepthPruned", "DepthPrunedMedusa", "DeepCache", "DeepCachePruned", "Sparse", "LCM"]})

    min_duration: Field = Field(
        "min_duration",
        "Minimum test duration",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS)

    max_duration: Field = Field(
        "max_duration",
        "Maximum test duration",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS)

    min_query_count: Field = Field(
        "min_query_count",
        "Minimum number of queries in test",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS)

    max_query_count: Field = Field(
        "max_query_count",
        "Maximum number of queries in test",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS)

    qsl_rng_seed: Field = Field(
        "qsl_rng_seed",
        "Seed for RNG that specifies which QSL samples are chosen for performance set and the order in which samples are processed in AccuracyOnly mode",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS)

    sample_index_rng_seed: Field = Field(
        "sample_index_rng_seed",
        "Seed for RNG that specifies order in which samples from performance set are included in queries",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS)

    use_spin_wait: Field = Field(
        "use_spin_wait",
        "Use spin waiting for LWIS. Recommended for single stream",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS)

    # TODO: I think 'test_run' needs to be handled specially, but not sure.
    test_run: Field = Field(
        "test_run",
        "If set, will set min_duration to 1 minute (60000ms). For Offline and Server, min_query_count is set to 1.",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS)

    logfile_suffix: Field = Field(
        "logfile_suffix",
        "Specify the filename suffix for the LoadGen log files",
        value_type=str,
        supported_actions=_HARNESS_ACTIONS)

    logfile_prefix_with_datetime: Field = Field(
        "logfile_prefix_with_datetime",
        "Prefix filenames for LoadGen log files",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS)

    log_copy_detail_to_stdout: Field = Field(
        "log_copy_detail_to_stdout",
        "Copy LoadGen detailed logging to stdout",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS)

    disable_log_copy_summary_to_stdout: Field = Field(
        "disable_log_copy_summary_to_stdout",
        "Disable copy LoadGen summary logging to stdout",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS)

    log_mode: Field = Field(
        "log_mode",
        "Logging mode for Loadgen",
        value_type=str,
        supported_actions=_HARNESS_ACTIONS,
        argparse_opts={"choices": ["AsyncPoll", "EndOfTestOnly", "Synchronous"]})

    log_mode_async_poll_interval_ms: Field = Field(
        "log_mode_async_poll_interval_ms",
        "Specify the poll interval for asynchrounous logging",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS)

    log_enable_trace: Field = Field(
        "log_enable_trace",
        "Enable trace logging",
        value_type=bool)

    use_triton: Field = Field(
        "use_triton",
        "Use Triton harness",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_harnesses=(HarnessType.Triton, HarnessType.TritonUnified),
        required=True)

    preferred_batch_size: Field = Field(
        "preferred_batch_size",
        "Preferred batch sizes",
        value_type=str)

    max_queue_delay_usec: Field = Field(
        "max_queue_delay_usec",
        "Set max queuing delay for Triton in usec.",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_harnesses=(HarnessType.Triton, HarnessType.TritonUnified))

    instance_group_count: Field = Field(
        "instance_group_count",
        "Set number of instance groups on each GPU.",
        value_type=int)

    request_timeout_usec: Field = Field(
        "request_timeout_usec",
        "Set the timeout for every request in usec.",
        value_type=int)

    buffer_manager_thread_count: Field = Field(
        "buffer_manager_thread_count",
        "The number of threads used to accelerate copies and other operations required to manage input and output tensor contents.",
        value_type=int,
        default=0)

    gather_kernel_buffer_threshold: Field = Field(
        "gather_kernel_buffer_threshold",
        "Set the threshold number of buffers for triton to use gather kernel to gather input data. 0 disables the gather kernel",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_harnesses=(HarnessType.Triton, HarnessType.TritonUnified),
        default=0)

    batch_triton_requests: Field = Field(
        "batch_triton_requests",
        "Send a batch of query samples to triton instead of single query at a time",
        supported_actions=_HARNESS_ACTIONS,
        supported_harnesses=(HarnessType.Triton, HarnessType.TritonUnified),
        value_type=bool)

    output_pinned_memory: Field = Field(
        "output_pinned_memory",
        "Use pinned memory when data transfer for output is between device mem and non-pinned sys mem",
        supported_actions=_HARNESS_ACTIONS,
        supported_harnesses=(HarnessType.Triton, HarnessType.TritonUnified),
        value_type=bool)

    use_concurrent_harness: Field = Field(
        "use_concurrent_harness",
        "Use multiple threads for batching and triton issue while using the triton harness",
        supported_actions=_HARNESS_ACTIONS,
        supported_harnesses=(HarnessType.Triton, HarnessType.TritonUnified),
        value_type=bool)

    num_concurrent_batchers: Field = Field(
        "num_concurrent_batchers",
        "Number of threads that will batch samples to form triton requests. Only used when the concurrent triton harness is used",
        supported_actions=_HARNESS_ACTIONS,
        supported_harnesses=(HarnessType.Triton, HarnessType.TritonUnified),
        value_type=int,
        default=1)

    num_concurrent_issuers: Field = Field(
        "num_concurrent_issuers",
        "Number of threads that will issue requests to triton. Only used when the concurrent triton harness is used.",
        supported_actions=_HARNESS_ACTIONS,
        supported_harnesses=(HarnessType.Triton, HarnessType.TritonUnified),
        value_type=int,
        default=1
    )

    dla_num_batchers: Field = Field(
        "dla_num_batchers",
        "Number of threads that will batch samples to form triton requests. Only used when the concurrent DLA triton harness is used",
        supported_actions=_HARNESS_ACTIONS,
        supported_harnesses=(HarnessType.Triton, HarnessType.TritonUnified),
        supported_systems=(SystemClassifications.is_soc,),
        value_type=int,
    )

    dla_num_issuers: Field = Field(
        "dla_num_issuers",
        "Number of threads that will issue requests to triton. Only used when the concurrent DLA triton harness is used.",
        supported_actions=_HARNESS_ACTIONS,
        supported_harnesses=(HarnessType.Triton, HarnessType.TritonUnified),
        supported_systems=(SystemClassifications.is_soc,),
        value_type=int,
    )

    # Many of these metrics have the scenario explicitly in the name because these are the names that Loadgen expects.
    server_target_qps_adj_factor: Field = Field(
        "server_target_qps_adj_factor",
        "Adjustment factor for target QPS used for server scenario",
        value_type=float,
        supported_actions=_HARNESS_ACTIONS,
        supported_scenarios=(Scenario.Server,))

    server_target_qps: Field = Field(
        "server_target_qps",
        "Target QPS used for server scenario",
        value_type=float,
        supported_actions=_HARNESS_ACTIONS,
        supported_scenarios=(Scenario.Server,))

    server_target_latency_ns: Field = Field(
        "server_target_latency_ns",
        "Desired latency constraint for server scenario",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_scenarios=(Scenario.Server,))

    server_target_latency_percentile: Field = Field(
        "server_target_latency_percentile",
        "Desired latency percentile constraint for server scenario",
        value_type=float,
        supported_actions=_HARNESS_ACTIONS,
        supported_scenarios=(Scenario.Server,))

    server_num_issue_query_threads: Field = Field(
        "server_num_issue_query_threads",
        "Number of IssueQuery threads to use for Loadgen in Server scenario",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_scenarios=(Scenario.Server,))

    schedule_rng_seed: Field = Field(
        "schedule_rng_seed",
        "Seed for RNG that affects the poisson arrival process in server scenario",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_scenarios=(Scenario.Server,))

    accuracy_log_rng_seed: Field = Field(
        "accuracy_log_rng_seed",
        "Affects which samples have their query returns logged to the accuracy log in performance mode.",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS)

    single_stream_expected_latency_ns: Field = Field(
        "single_stream_expected_latency_ns",
        "Inverse of desired target QPS",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_scenarios=(Scenario.SingleStream,))

    single_stream_target_latency_percentile: Field = Field(
        "single_stream_target_latency_percentile",
        "Desired latency percentile constraint for single stream scenario",
        value_type=float,
        supported_actions=_HARNESS_ACTIONS,
        supported_scenarios=(Scenario.SingleStream,))

    multi_stream_expected_latency_ns: Field = Field(
        "multi_stream_expected_latency_ns",
        "Expected latency to process a query with multiple Samples, in nanoseconds",
        value_type=int,
        required=True,
        supported_actions=_HARNESS_ACTIONS,
        supported_scenarios=(Scenario.MultiStream,))

    multi_stream_target_latency_percentile: Field = Field(
        "multi_stream_target_latency_percentile",
        "Desired latency percentile to report as a performance metric, for multi stream scenario",
        value_type=float,
        required=True,
        supported_actions=_HARNESS_ACTIONS,
        supported_scenarios=(Scenario.MultiStream,))

    multi_stream_samples_per_query: Field = Field(
        "multi_stream_samples_per_query",
        "Number of samples bundled together as a single query (default: 8)",
        value_type=int,
        default=8,
        required=True,
        supported_actions=_HARNESS_ACTIONS,
        supported_scenarios=(Scenario.MultiStream,))

    offline_expected_qps: Field = Field(
        "offline_expected_qps",
        "Target samples per second rate for the SUT (Offline mode)",
        value_type=float,
        supported_actions=_HARNESS_ACTIONS,
        supported_scenarios=(Scenario.Offline,))

    sample_partition_path: Field = Field(
        "sample_partition_path",
        "Path to sample partition file in npy format.",
        value_type=str,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.DLRMv2,))

    embedding_weights_on_gpu_part: Field = Field(
        "embedding_weights_on_gpu_part",
        """Percentage of the embedding weights to store on GPU. Lower this if your GPU has lower VRAM. Default: 1.0
        (100% of embedding weights on GPU)""",
        value_type=float,
        supported_benchmarks=(Benchmark.DLRMv2,),
        supported_systems=(SystemClassifications.gpu_based,))

    use_single_qsl: Field = Field(
        "qsl_numa_override",
        """Designate NUMA node(s) each QSL maps to; this overrides numa config for QSL""",
        value_type=str,
        supported_benchmarks=(Benchmark.DLRMv2,),
        supported_systems=(SystemClassifications.gpu_based,))

    num_staging_threads: Field = Field(
        "num_staging_threads",
        "Number of staging threads in DLRMv2 BatchMaker",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.DLRMv2,))

    num_staging_batches: Field = Field(
        "num_staging_batches",
        "Number of staging batches in DLRMv2 BatchMaker",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.DLRMv2,))

    max_pairs_per_staging_thread: Field = Field(
        "max_pairs_per_staging_thread",
        "Maximum pairs to copy in one BatchMaker staging thread",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.DLRMv2,))

    gpu_num_bundles: Field = Field(
        "gpu_num_bundles",
        "Number of event+buffer bundles per GPU for DLRMv2 (default: 2)",
        value_type=int,
        default=2,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.DLRMv2,),
        supported_systems=(SystemClassifications.gpu_based,))

    check_contiguity: Field = Field(
        "check_contiguity",
        "Check if inputs are already contiguous in QSL to avoid copying",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.DLRMv2,))

    embeddings_path: Field = Field(
        "embeddings_path",
        "Path to DLRMv2 embeddings in npy format.",
        value_type=str,
        default='/home/mlperf_inf_dlrmv2/model/embedding_weights/',
        supported_benchmarks=(Benchmark.DLRMv2,))

    bot_mlp_precision: Field = Field(
        'bot_mlp_precision',
        'Precision to run bot-mlp in',
        value_type=str,
        supported_benchmarks=(Benchmark.DLRMv2,))

    embeddings_precision: Field = Field(
        'embeddings_precision',
        'Precision to run embeddings in',
        value_type=str,
        supported_benchmarks=(Benchmark.DLRMv2,))

    interaction_op_precision: Field = Field(
        'interaction_op_precision',
        'Precision to run interaction-op in',
        value_type=str,
        supported_benchmarks=(Benchmark.DLRMv2,))

    top_mlp_precision: Field = Field(
        'top_mlp_precision',
        'Precision to run top-mlp in',
        value_type=str,
        supported_benchmarks=(Benchmark.DLRMv2,))

    final_linear_precision: Field = Field(
        'final_linear_precision',
        'Precision to run final-linear in',
        value_type=str,
        supported_benchmarks=(Benchmark.DLRMv2,))

    use_jemalloc: Field = Field(
        "use_jemalloc",
        "Use libjemalloc.so.2 as the malloc(3) implementation",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS)

    bert_opt_seqlen: Field = Field(
        "bert_opt_seqlen",
        "The opt_shape for BERT provided to TRT builder in the optimization profile.",
        value_type=int,
        supported_actions=(Action.GenerateEngines,),
        supported_benchmarks=(Benchmark.BERT,))

    graphs_max_seqlen: Field = Field(
        "graphs_max_seqlen",
        "The max sequence length for BERT when capturing CUDA Graphs.",
        value_type=int,
        supported_actions=(Action.GenerateEngines,),
        supported_benchmarks=(Benchmark.BERT,))

    graph_specs: Field = Field(
        "graph_specs",
        """Specify a comma separated list of (maxSeqLen, min totSeqLen, max totSeqLen, step size) for CUDA graphs to be
        captured.""",
        value_type=str,
        supported_actions=(Action.GenerateEngines,),
        supported_benchmarks=(Benchmark.BERT,))

    soft_drop: Field = Field(
        "soft_drop",
        """Inverse percentage of the longest BERT sequences to drop softly in Server scenario by deferring to the end.
        When soft_drop=1.0, no sequence is dropped. When soft_drop=0.992, the top 0.8% longest sequences are dropped
        softly, i.e. their inferences are pushed to the end of the test.""",
        value_type=float,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.BERT,))

    use_fp8: Field = Field(
        "use_fp8",
        "Enable FP8 sub-part for the network. The TRT network might still be in the original precision",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS + (Action.GenerateEngines,),
        supported_systems=(SystemClassifications.gpu_based,),
        supported_benchmarks=(Benchmark.BERT, Benchmark.GPTJ, Benchmark.LLAMA2, Benchmark.Mixtral8x7B,))

    trtllm_batching_mode: Field = Field(
        "trtllm_batching_mode",
        "Batching mode of TRTLLM batch manager. Choices are V1 and IFB",
        value_type=str,
        supported_actions=_HARNESS_ACTIONS + (Action.GenerateEngines,),
        supported_systems=(SystemClassifications.gpu_based,),
        supported_benchmarks=(Benchmark.GPTJ, Benchmark.LLAMA2, Benchmark.Mixtral8x7B,))

    trtllm_batch_sched_policy: Field = Field(
        "trtllm_batch_sched_policy",
        "TRTLLM batch scheduling policy. Options are max_util and no_evict",
        value_type=str,
        supported_actions=_HARNESS_ACTIONS + (Action.GenerateEngines,),
        supported_systems=(SystemClassifications.gpu_based,),
        supported_benchmarks=(Benchmark.GPTJ, Benchmark.LLAMA2, Benchmark.Mixtral8x7B,))

    enable_sort: Field = Field(
        "enable_sort",
        "Sort QSL samples before inference.",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_systems=(SystemClassifications.gpu_based,),
        supported_benchmarks=(Benchmark.GPTJ, Benchmark.LLAMA2, Benchmark.Mixtral8x7B,))

    num_sort_segments: Field = Field(
        "num_sort_segments",
        "The number of segments to divide the awiting queries before sorting.",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_systems=(SystemClassifications.gpu_based,),
        supported_benchmarks=(Benchmark.GPTJ, Benchmark.LLAMA2, Benchmark.Mixtral8x7B,))

    tensor_parallelism: Field = Field(
        "tensor_parallelism",
        "Tensor Parallelism value for LLM",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS + (Action.GenerateEngines,),
        supported_systems=(SystemClassifications.gpu_based,),
        supported_benchmarks=(Benchmark.GPTJ, Benchmark.LLAMA2, Benchmark.Mixtral8x7B,))

    pipeline_parallelism: Field = Field(
        "pipeline_parallelism",
        "Pipeline Parallelism value for LLM",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS + (Action.GenerateEngines,),
        supported_systems=(SystemClassifications.gpu_based,),
        supported_benchmarks=(Benchmark.GPTJ, Benchmark.LLAMA2, Benchmark.Mixtral8x7B,))

    gpu_rank_map: Field = Field(
        "gpu_rank_map",
        "GPU ID mapping with respect to MPI rank for LLM using TP/PP",
        value_type=str,
        supported_actions=_HARNESS_ACTIONS,
        supported_systems=(SystemClassifications.gpu_based,),
        supported_benchmarks=(Benchmark.LLAMA2, Benchmark.Mixtral8x7B,))

    kvcache_free_gpu_mem_frac: Field = Field(
        "kvcache_free_gpu_mem_frac",
        "For TRTLLM batcher, the fraction of GPU memory that can be occupied by the KV-cache",
        value_type=float,
        supported_actions=_HARNESS_ACTIONS,
        supported_systems=(SystemClassifications.gpu_based,),
        supported_benchmarks=(Benchmark.GPTJ, Benchmark.LLAMA2, Benchmark.Mixtral8x7B,))

    max_num_tokens: Field = Field(
        "max_num_tokens",
        "Maximum number of tokens engine could generate",
        value_type=int,
        supported_actions=(Action.GenerateEngines,),
        supported_systems=(SystemClassifications.gpu_based,),
        supported_benchmarks=(Benchmark.GPTJ, Benchmark.LLAMA2, Benchmark.Mixtral8x7B,))

    llm_gen_config_path: Field = Field(
        "llm_gen_config_path",
        "The path to the json files storing the generation configs.",
        value_type=str,
        supported_actions=_HARNESS_ACTIONS,
        supported_systems=(SystemClassifications.gpu_based,),
        supported_benchmarks=(Benchmark.GPTJ, Benchmark.LLAMA2, Benchmark.Mixtral8x7B,))

    use_token_latencies: Field = Field(
        "use_token_latencies",
        "The path to the json files storing the generation configs.",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_systems=(SystemClassifications.gpu_based,),
        supported_benchmarks=(Benchmark.GPTJ, Benchmark.LLAMA2, Benchmark.Mixtral8x7B,))

    vboost_slider: Field = Field(
        "vboost_slider",
        "Control clock-propagation ratios between GPC-XBAR. Look at `nvidia-smi boost-slider --vboost`.",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_systems=(SystemClassifications.gpu_based,))

    numa_config: Field = Field(
        "numa_config",
        """NUMA settings: GPU and CPU cores for each NUMA node as a string with the following convention:

        ```
        [Node 0 config]&[Node 1 config]&[Node 2 config]&...
        ```

        Each `[Node n config]` can be configured as `[GPU ID(s)]:[CPU ID(s)]`.
        Each `ID(s)` can be single digit, comma-separated digits, or digits with dash.
        ex.
            "numa_config": "3:0-15,64-79&2:16-31,80-95&1:32-47,96-111&0:48-63,112-127"

        In this example, example `3:0-15,64-79` means GPU 3 and CPU 0,1,2,...15,64,65,66,...79 are in the same node,
        and since this key is the very first of elements connected with &, they are in node 0.""",
        value_type=str,
        supported_actions=_HARNESS_ACTIONS,
        supported_systems=(SystemClassifications.multi_gpu,))

    nms_type: Field = Field(
        "nms_type",
        "Use EfficientNMS/NMSOptPlugin/RetinaNetNMSPVATRT plugin for RetinaNet NMS layers, supported value: {'efficientnms', 'nmsopt', 'nmspva'}",
        value_type=str,
        supported_actions=_HARNESS_ACTIONS + (Action.GenerateEngines,),
        supported_benchmarks=(Benchmark.Retinanet,))

    sdxl_batcher_time_limit: Field = Field(
        "sdxl_batcher_time_limit",
        "SDXL harness server scenario batcher time out threashold in seconds",
        value_type=float,
        supported_actions=_HARNESS_ACTIONS,
        supported_scenarios=(Scenario.Server,),
        supported_systems=(SystemClassifications.gpu_based,),
        supported_benchmarks=(Benchmark.SDXL,))

    sdxl_shallow_unet_steps: Field = Field(
        "sdxl_shallow_unet_steps",
        "SDXL number of shallow unet steps to run after a deep unet",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_systems=(SystemClassifications.gpu_based,),
        supported_benchmarks=(Benchmark.SDXL,))


def get_applicable_fields(action: Optional[Action], benchmark: Benchmark, scenario: Scenario, system: System,
                          workload_setting: WorkloadSetting) -> Tuple[List[Field], List[Field]]:
    """
    Gets the applicable fields for the workload from Fields.

    Args:
        action (Optional[Action]):
            The Action of the workload. Can be None.
        benchmark (Benchmark):
            The Benchmark of the workload
        scenario (Scenario):
            The Scenario of the workload
        system (System):
            The System running the workload
        workload_setting (WorkloadSetting):
            The WorkloadSetting field of the config to get fields for

    Returns:
        List[Field]: A list of the required fields for the workload to run
        List[Field]: A list of the optional (non-required) but supported fields for the workload to run
    """
    required = []
    optional = []
    for field in Fields:
        if field.value.supports(action, benchmark, scenario, system, workload_setting):
            if field.value.required:
                required.append(field.value)
            else:
                optional.append(field.value)
    return required, optional


def apply_helper_and_legacy_fields(benchmark_conf: Dict[str, Any],
                                   workload_setting: WorkloadSetting,
                                   system_name_override: Optional[str] = None) -> Dict[str, Any]:
    """Takes in a BenchmarkConfiguration in dictionary form and adds keys that are used either as helper values or to
    support legacy behavior.

    This function adds the following helper keys:
        - system_id: A string representing the ID of the system in the configuration
        - config_name: A unique, human-readable identifier for the config. Used for internal bookkeeping
        - workload_setting: The workload_setting this config is run for
        - optimization_level: Currently, always "plugin-enabled". This key is supposed to indicate the optimzation level
                              of the software for inference. Supports the planned "out-of-the-box" TensorRT submissions
                              that use TensorRT without any custom plugins.

    This function also adds or modifies the following keys for legacy behavior:
        - system_name: This key serves as an override so that users can change the system name used in submission files
                       (such as the measurements/ and results/ directories) via a command line flag. This feature was
                       used by some submission partners in the past so they could use naming that wasn't standard with
                       NVIDIA product name policies. This feature is not recommended for internal NVIDIA users, and is
                       considered legacy behavior.
        - config_ver: A shortname string that represents the WorkloadSetting. This is legacy behavior, as the feature
                      was replaced by WorkloadSettings.
        - accuracy_level: A convenience key. Either '99%' or '99.9%' to represent the accuracy target. This is legacy
                      behavior, as the feature was replaced by WorkloadSetting.accuracy_target. The CI/CD pipeline still
                      uses this, so it is still supported.
        - inference_server: A convenience key. String representation of the harness type. This is legacy behavior, as
                            the feature was replaced by WorkloadSetting.harness_type. The CI/CD pipeline still uses
                            this, so it is still supported.


    Args:
        benchmark_conf (Dict[str, Any]): Dictionary representing the benchmark config.
        workload_setting (WorkloadSetting): The workload setting the config is being run on.
        system_name_override (Optional[str]): If set, used as the system_name override.

    Returns:
        Dict[str, Any]: A dictionary representation of the BenchmarkConfiguration with all the fields added in.
    """
    # Helper keys
    system = benchmark_conf["system"]
    system_id = system.get_id()
    benchmark = benchmark_conf["benchmark"]
    scenario = benchmark_conf["scenario"]
    benchmark_conf["system_id"] = system_id
    benchmark_conf["config_name"] = f"{system_id}_{benchmark.valstr()}_{scenario.valstr()}"
    benchmark_conf["workload_setting"] = workload_setting
    benchmark_conf["optimization_level"] = "plugin-enabled"
    if len(system.accelerator_conf.get_accelerators()) == 0:
        raise NotImplementedError(f"CPU-only system is not supported")

    # For Mitten
    # Prior to TRT 8.6+, one optimization profile can only be linked to one set of bindings. In most harness cases,
    # there is one binding per copy stream. Set num_profiles = gpu_copy_streams.
    benchmark_conf["num_profiles"] = benchmark_conf.get("gpu_copy_streams", 1)

    # Legacy keys
    if system_name_override is not None:
        benchmark_conf["system_name"] = system_name_override
    workload_id = workload_setting.shortname()
    benchmark_conf["config_ver"] = workload_id
    benchmark_conf["accuracy_level"] = "99%" if workload_setting.accuracy_target == AccuracyTarget.k_99 else "99.9%"
    benchmark_conf["inference_server"] = str(workload_setting.harness_type.value)
    return benchmark_conf


def get_effective_values(benchmark_conf: Dict[str, Any],
                         action: Optional[Action],
                         workload_setting: WorkloadSetting) -> Dict[str, Any]:
    """
    Filters the benchmark_conf for applicable values, and then applies overrides. The override priority is as follows
    (from highest priority to default):
        1. Command line flags (i.e. RUN_ARGS)
        2. The value that exists in benchmark_conf already
        3. The default value for the field (i.e. Field.default)

    Args:
        benchmark_conf (Dict[str, Any]): The base benchmark config as a dict
        action (Optional[Action]): The Action of the workload. Can be None.
        workload_setting (WorkloadSetting): The WorkloadSetting field of the config to get fields for

    Returns:
        Dict[str, Any]: The new dict with overriden values
    """
    # Filter out the base config for applicable fields
    req, opt = get_applicable_fields(action,
                                     benchmark_conf["benchmark"],
                                     benchmark_conf["scenario"],
                                     benchmark_conf["system"],
                                     workload_setting)
    all_fields = req + opt

    # Build the base configuration with default values
    filtered = dict()
    for field in all_fields:
        name = field.name
        # Use the value in benchmark_conf if it exists, otherwise default
        if name in benchmark_conf:
            filtered[name] = benchmark_conf[name]
        else:
            if field.default is not None:
                filtered[name] = field.default

    # Find any command line overrides
    parser = ArgumentParser(allow_abbrev=False)  # By default, argparse auto-adds a shortening (--foo to -f)
    for field in all_fields:
        field.add_to_argparser(parser)
    overrides = vars(parser.parse_known_args()[0])

    bs_pattern = r'^.+:\d+(?:,.+:\d+)*$'  # required format is component1:component_bs1,component2:component_bs2,...
    for name, value in overrides.items():
        # Unset values (None) and unset store_true values (False) are both false-y
        if value:
            # Parse user batch size
            if name in ("gpu_batch_size", "dla_batch_size"):
                if not re.fullmatch(bs_pattern, value):
                    raise ValueError(f"Specifying gpu_batch_size or dla_batch_size in RUN_ARGs must follow the format of component1:component_bs1,component2:component_bs2,... Wrong input: {value}")
                bs_pairs = value.split(',')
                # Split each pair by the colon and construct the dictionary
                batch_size_dict = {}
                for pair in bs_pairs:
                    component, component_batch_size = pair.split(':')
                    batch_size_dict[component] = int(component_batch_size)
                filtered[name] = batch_size_dict
            else:
                filtered[name] = value

    # Sort alphabetically
    config_dict = dict(sorted(filtered.items()))

    # Check type
    for field in all_fields:
        name = field.name
        if name in config_dict:
            try:
                check_type(config_dict[name], field.value_type)
            except TypeCheckError:
                raise TypeError(f"{name}: {config_dict[name]} does not match expected type {field.value_type}")

    return config_dict


@unique
class MainArgs(Enum):
    """
    Defines a set of known Fields used as CLI arguments for main.py. Note that Fields specified in this Enum will ignore
    all `Field.supported_*` attributes, since these values will not be known until the appropriate action is determined
    from MainArgs.action.
    """

    action: Field = Field(
        "action",
        "The phase in the benchmarking or submission process to run.",
        value_type=str,
        argparse_opts={
            "choices": Action.as_strings(),
            "required": True,
        })

    benchmarks: Field = Field(
        "benchmarks",
        "The names of benchmarks to run the current phase on. " +
        "If specifying multiple benchmarks, use a comma,separated,list of names (i.e. resnet50,bert,dlrm) " +
        "Default: Runs all benchmarks with valid configurations for the current system.",
        value_type=str)

    scenarios: Field = Field(
        "scenarios",
        "The names of scenarios to run the current phase for. " +
        "If specifying multiple scenarios, use a comma,separated,list of names (i.e. offline,server) " +
        "Default: Runs all scenarios with valid configurations for the current system.",
        value_type=str)

    gpu_only: Field = Field(
        "gpu_only",
        "Only used on SoC machines where default behavior is to use the iGPU and both DLAs. " +
        "If set, then the current run will only use the iGPU for the specified action.",
        value_type=bool)

    no_gpu: Field = Field(
        "no_gpu",
        "Only used on SoC machines where default behavior is to use the iGPU and both DLAs. " +
        "If set, then the current run will only use the DLA cores specified without the iGPU. " +
        "Default: DLA Core 0",
        value_type=bool)

    no_child_process: Field = Field(
        "no_child_process",
        "Generate engines in main process instead of child processes. Useful when running profilers or gdb " +
        "when debugging issues with generate_engines.",
        value_type=bool)

    power: Field = Field(
        "power",
        "Measure power during this harness run. Note that this does NOT set power_setting to MaxQ.",
        value_type=bool)

    # power_limit is specially handled, since it must be known *before* the action is run, when the ScopedMPS is
    # created. Hence, power_limit (which should be overridable), must be applied as an override at the initial main.py
    # level, not at the BenchmarkConfiguration level.
    # Server power args
    power_limit: Field = Fields.power_limit.value
    cpu_freq: Field = Fields.cpu_freq.value
    # SoC power args
    soc_gpu_freq: Field = Fields.soc_gpu_freq.value
    soc_dla_freq: Field = Fields.soc_dla_freq.value
    soc_cpu_freq: Field = Fields.soc_cpu_freq.value
    soc_emc_freq: Field = Fields.soc_emc_freq.value
    orin_num_cores: Field = Fields.orin_num_cores.value

    profile: Field = Field(
        "profile",
        "[INTERNAL ONLY] Select profiler to use.",
        value_type=str,
        argparse_opts={"choices": ["nsys", "nvprof", "ncu", "pic-c"]})

    audit_test: Field = Field(
        "audit_test",
        "The audit test to run, if an audit-related action is chosed.",
        value_type=str,
        argparse_opts={"choices": ["TEST01", "TEST04", "TEST05", "TEST06"]})

    no_audit_verify: Field = Field(
        "no_audit_verify",
        "If set, skip the verification step for the audit harness. Ignored if not running audit harness.",
        value_type=bool
    )

    system_name: Field = Field(
        "system_name",
        "Overrides the 'system name', which is generated from the detected system by default. " +
        "In most cases, it is better to set 'system_name' directly in the BenchmarkConfiguration " +
        "instead of using this field to keep the system_name consistent across runs.",
        value_type=str)

    harness_type: Field = Field(
        "harness_type",
        "Selects which harness to use during the run_harness phase. Note that this should also " +
        "be specified during the generate_engines phase, as changing harnesses may also change " +
        "engine-generation parameters. Use 'auto' to use the default harness (LWIS or custom). " +
        "Default: 'auto'",
        value_type=str,
        default="auto",
        argparse_opts={
            "choices": ["auto"] + HarnessType.as_strings(),
            "default": "auto",
        })

    accuracy_target: Field = Field(
        "accuracy_target",
        "Selects which accuracy target to compare the accuracy results against. Note that the " +
        "'high accuracy' (99.9%% of FP32) target is only available on BERT, DLRMv2, and 3D-UNET.",
        value_type=float,
        default=AccuracyTarget.k_99.value,
        argparse_opts={
            "choices": [at.value for at in AccuracyTarget],
            "default": AccuracyTarget.k_99.value,
        })

    power_setting: Field = Field(
        "power_setting",
        "Selects which power setting the harness is running on. Note that this setting does NOT " +
        "actually set the power setting itself, but instead is just used to select the appropriate " +
        "BenchmarkConfiguration. The power setting must be set with the appropriate commands " +
        "before this program is run.",
        value_type=str,
        default=PowerSetting.MaxP.value.name,
        argparse_opts={
            "choices": PowerSetting.as_strings(),
            "default": PowerSetting.MaxP.value.name,
        })

    model_opt: Field = Field(
        "model_opt",
        "Selects which model optimization method to use for this run. Note that a value of this "
        "setting may only be applicable to one or more benchmarks. By default, we apply no model opt.",
        value_type=str,
        default=ModelOpt.NoOpt,
        argparse_opts={
            "choices": ModelOpt.as_strings(),
            "default": ModelOpt.NoOpt.id(),
        })

    config_ver: Field = Field(
        "config_ver",
        "[LEGACY OPTION] Specifies workload settings via the config_ver IDs from the old config.json " +
        "style configs. Using this flag will override the --harness_type, --accuracy_target, and " +
        "--power_setting flags.",
        value_type=str)
