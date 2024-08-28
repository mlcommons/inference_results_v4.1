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
import queue
from collections import namedtuple
import time
import argparse
import threading
import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional, Union
import json
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import importlib.util
from code.harness.harness_triton_llm.frontend import TritonSutGrpcStreamFrontend, get_client
from code.harness.harness_triton_llm.utils import construct_triton_input_data, get_gptj_triton_inputs, LlmConfig, get_llm_gen_config
from code.harness.harness_triton_llm.dataset import LlmDataset

try:
    import nvidia_smi
    nvidia_smi.nvmlInit()
except:
    logging.warning("nvidia-ml-py3 not installed. Utilization stats will not be logged.")


QuerySample = namedtuple('QuerySample', 'id, index')


class SequentialBenchmark:
    """
    a batcher to mimic the MLPerf Loadgen
    Will produce inference requests at a given rate, and goes sequentially from first sequence to last
    """

    def __init__(self, target_qps: float, performance_sample_count: int):
        self.target_qps = target_qps
        self.performance_sample_count = performance_sample_count
        self.report_token_queue = queue.Queue()
        self.gatherer = threading.Thread(target=self.gather_token)
        self.gatherer.start()
        self.stats = [['sample_id', 'is_final', 'token', 'ts'], ]

    def start(self):
        sleep_time = 1 / self.target_qps
        logging.info(f"{self.performance_sample_count} samples will be scheduled, with {sleep_time} sec break between consecutive requests made")
        for i in range(self.performance_sample_count):
            frontend.dispatch_query_samples([QuerySample(id=str(i), index=i)])
            time.sleep(sleep_time)
            if (i + 1) % 1000 == 0:
                logging.info(f"[SequentialBenchmark] Dispatched {i+1} samples")
        frontend.notify_dispatch_done()

    def gather_token(self):
        while True:
            tok = self.report_token_queue.get()
            if tok is None:
                self.report_token_queue.task_done()
                break
            tok.append(time.time())
            self.stats.append(tok)

            self.report_token_queue.task_done()

    def _dump_stats(self):
        with open('seq_stats.json', 'w') as f:
            json.dump(self.stats, f)
        logging.info(f"[SequentialBenchmark] dumped stats")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--llm_gen_config_path", help="Path to generation_config.json")
    parser.add_argument("--tensor_path", type=str, help="(comma separated) path(s) to the dataset tensor file")
    parser.add_argument("--performance_sample_count", type=int, default=5000, help="Number of samples to run benchmark on")
    parser.add_argument("--num_clients", type=int, default=4, help="Number of grpc clients to initialize per GPU")
    parser.add_argument("--target_qps", type=float, default=11.5, help="Number of queries to sample per second.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--verbose_frontend", action="store_true", help="Make triton client verbose", default=True)

    args = parser.parse_args()
    return args


def get_llm_gen_config(model_name: str, scenario: str, llm_gen_config_path: str):
    with open(llm_gen_config_path) as f:
        vals = json.load(f)['generation_config']
        llm_config = LlmConfig(
            model_name=model_name,
            min_output_len=vals['min_output_len'],
            max_output_len=vals['max_output_len'],
            beam_width=vals['runtime_beam_width'],
            end_token_id=vals['eos_token_id'],
            pad_token_id=vals['eos_token_id'],
            streaming=vals['streaming'] and scenario != "Offline",
        )
        return llm_config


if __name__ == "__main__":
    args = parse_args()
    model = "llama2-70b"
    scenario = "Server"

    llm_config = get_llm_gen_config(model, scenario, args.llm_gen_config_path)
    input_ids, input_lens = args.tensor_path.split(',')
    ds = LlmDataset(path_input_ids=input_ids, path_input_lens=input_lens)
    model_name_prefix = llm_config.model_name.lower() + '-' + scenario.lower()

    dispatcher = SequentialBenchmark(target_qps=args.target_qps, performance_sample_count=args.performance_sample_count)
    frontend = TritonSutGrpcStreamFrontend(verbose=args.verbose_frontend, report_token_queue=dispatcher.report_token_queue, num_clients_per_gpu=args.num_clients,
                                           dataset=ds, llm_config=llm_config, model_prefix=model_name_prefix, num_gpus=args.num_gpus)

    dispatcher.start()
