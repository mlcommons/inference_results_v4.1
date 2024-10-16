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
from code.common.harness import BaseBenchmarkHarness
from code.common.triton.base_config import G_TRITON_BASE_CONFIG
from code.common import logging, args_to_string
from code.common.constants import Benchmark, Scenario
import code.common.arguments as common_args
import os


class TritonLlmHarness(BaseBenchmarkHarness):

    def __init__(self, args, benchmark):
        super().__init__(args, benchmark)
        self.model_store_path = os.path.abspath("/work/build/triton_model_repo")
        self.tensorrt_lib_path = os.path.abspath("/work/build/triton-inference-server/out/tensorrtllm/install/backends")
        self.model_name = self._get_model_name(args)
        self.model_version = "1"
        custom_args = [
            "llm_gen_config_path",
            "model_repo",
            "num_gpus",
            "use_token_latencies"
        ]
        self.flag_builder_custom_args = common_args.LOADGEN_ARGS + common_args.SHARED_ARGS + custom_args
        self.num_gpus = self.args['system'].accelerator_conf.num_gpus()

    def _get_model_name(self, config):
        benchmark = config["benchmark"].valstr().lower()
        scenario = config["scenario"].valstr().lower()
        return "{}-{}".format(benchmark, scenario)

    def _get_harness_executable(self):
        return "code/harness/harness_triton_llm/main.py"

    def _construct_terminal_command(self, argstr):
        cmd = f"{self.executable.replace('code/harness/harness_triton_llm/main.py', 'python3 -m code.harness.harness_triton_llm.main')} {argstr}"
        return cmd

    def _get_engine_dirpath(self, device_type, batch_size):
        dirpath = os.path.join(self._get_engine_fpath(device_type, None, batch_size), os.pardir)
        return os.path.abspath(dirpath)

    def _append_config_ver_name(self, system_name):
        system_name += "_Triton"
        return super()._append_config_ver_name(system_name)

    def _build_custom_flags(self, flag_dict):
        argstr = args_to_string(flag_dict)
        argstr = argstr + " --scenario " + self.scenario.valstr()
        argstr = argstr + " --model " + self.name
        argstr = argstr + " --model_repo " + self.model_store_path
        argstr = argstr + " --num_gpus " + str(self.num_gpus)
        if self.scenario.valstr().lower() == "server":
            argstr = argstr + " --num_clients_per_gpu 4"  # TODO: Parse from config
        return argstr

    def run_harness(self, flag_dict=None, skip_generate_measurements=False):
        self._setup_triton_model_repo(flag_dict)
        return super().run_harness(flag_dict, skip_generate_measurements)

    def _setup_triton_model_repo(self, flag_dict, beam_width, decoupled):
        triton_model_name_prefix = self.name.lower() + '-' + self.scenario.valstr().lower()
        for gpu_idx in range(self.num_gpus):
            triton_model_name = triton_model_name_prefix + '-' + str(gpu_idx)
            model_dir = os.path.join(self.model_store_path, triton_model_name, self.model_version)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            config_file_path = os.path.join(self.model_store_path, triton_model_name, "config.pbtxt")
            engine_file_name = self._get_engine_dirpath(device_type=None, batch_size=flag_dict["gpu_batch_size"])
            with open(config_file_path, 'w') as f:
                f.write(G_TRITON_BASE_CONFIG.format(
                    model_name=triton_model_name,
                    is_decoupled=decoupled,
                    beam_width=beam_width,
                    engine_path=engine_file_name,
                    gpu_device_idx=gpu_idx))
