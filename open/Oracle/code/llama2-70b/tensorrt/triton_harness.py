# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
from code.common.triton.base_harness import TritonLlmHarness


class TritonLlamaHarness(TritonLlmHarness):
    def _get_engine_fpath(self, device_type, _, batch_size):
        assert self.name == "llama2-70b"
        return f"{self.engine_dir}/bs{batch_size}-{self.config_ver}-tp{self.tp_size}-pp{self.pp_size}/rank0.engine"

    def _setup_triton_model_repo(self, flag_dict):
        beam_width = 1
        decoupled = False
        if self.scenario.valstr().lower() == "server":
            decoupled = True

        super()._setup_triton_model_repo(flag_dict, beam_width=beam_width, decoupled=decoupled)
