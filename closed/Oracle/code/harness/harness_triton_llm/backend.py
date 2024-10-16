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
import subprocess
import logging

import socket
from contextlib import closing


# from https://stackoverflow.com/a/45690594/5076583
def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class TritonSutBackend:
    """
    The part of the triton SUT that is treated as a black box.
    Owns the tritonserver process.
    """

    def __init__(self,
                 binary_exec: str = "/work/build/triton-inference-server/out/opt/tritonserver/bin/tritonserver",
                 model_repo: str = "/work/triton-proj/triton-models",
                 model_name_prefix: str = "llama2-offline",
                 backend_dir: str = "/work/build/triton-inference-server/out/tensorrtllm/install/backends",
                 num_gpus: int = 1,
                 allow_http: bool = False,
                 allow_grpc: bool = True,
                 allow_metrics: bool = True):
        cmd = [binary_exec]
        cmd.append(f"--model-repository={model_repo}")
        cmd.append(f"--backend-directory={backend_dir}")
        cmd.append("--strict-readiness=true")
        cmd.append("--model-control-mode=explicit")

        def enable_service(service, is_enabled):
            if is_enabled:
                port = find_free_port()
                cmd.append(f"--{service}-port={port}")
            else:
                cmd.append(f"--allow-{service}=false")
                port = -1
            return port

        self.grpc_port = enable_service("grpc", allow_grpc)
        self.http_port = enable_service("http", allow_http)
        self.metrics_port = enable_service("metrics", allow_metrics)

        for gpu_id in range(num_gpus):
            cmd.append(f"--load-model={model_name_prefix}-{gpu_id}")

        logging.info(f"[TritonSutBackend] Executing command: {' '.join(cmd)}")

        self._server_proc = subprocess.Popen(cmd)

    def get_grpc_port(self):
        return self.grpc_port

    def get_http_port(self):
        return self.http_port

    def get_metrics_port(self):
        return self.metrics_port

    def __del__(self):
        self._server_proc.kill()
        logging.info("Finished destroying TritonSutBackend")
