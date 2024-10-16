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
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
import threading
import queue
from abc import ABC, abstractmethod
from typing import Callable
import time
import mlperf_loadgen as lg
import numpy as np
import logging
import json
import importlib.util
from typing import Union
import multiprocessing as mp

try:
    import nvidia_smi
    nvidia_smi.nvmlInit()
except:
    logging.warning("nvidia-ml-py3 not installed. Utilization stats will not be logged.")

from code.harness.harness_triton_llm.utils import LlmConfig, get_gptj_triton_inputs
from code.harness.harness_triton_llm.dataset import LlmDataset


def get_client(url: str, protocol: str = "http", concurrency: int = 192, verbose: bool = False):
    assert protocol in ["http", "grpc"]
    if protocol == "http":
        return httpclient.InferenceServerClient(url=url, concurrency=concurrency, verbose=verbose)
    else:
        return grpcclient.InferenceServerClient(url=url, verbose=verbose)


class ITritonSutFrontend(ABC):
    def __init__(self,
                 dataset: LlmDataset,
                 llm_config: LlmConfig,
                 triton_model_name: str,
                 report_loadgen_queue: Union[queue.Queue, mp.Queue, None] = None,
                 url: str = "0.0.0.0:8001",
                 verbose: bool = False,
                 log_stats: bool = False,
                 num_clients_per_gpu: int = 1,
                 triton_batch_size: int = 1):
        assert num_clients_per_gpu >= 1, "Number of clients for a single GPU must be >= 1 (4 for Server, 1 for Offline)"
        assert triton_batch_size == 1, "BS>1 is not supported in Triton for now, check Jira TRTLLM-298"
        self.report_loadgen_queue = report_loadgen_queue  # report QuerySampleResponses to master process

        if log_stats:
            logging.info("Stat logging run for triton harness detected. This may harm performance.")
        self.verbose = verbose
        self.log_stats = log_stats
        self.url = url
        self.clients = []
        self.dispatch_queues = []
        self.dispatchers = []
        self.num_dispatchers = num_clients_per_gpu
        self._ds = dataset
        self.llm_config = llm_config
        self.triton_batch_size = triton_batch_size
        self.num_queries_dispatched = 0
        self.num_total_queries_dispatched = 0
        self.queries_responded = 0
        self.triton_model_name = triton_model_name
        self.dispatching_lock = threading.Lock()  # acquired when dispatch_query_samples_called, to guard against premature termination

        for _ in range(self.num_dispatchers):
            self.clients.append(get_client(url=url, protocol="grpc", verbose=False))

        self.wait_for_server_readiness(self.clients[0].is_server_ready)

        for dispatch_idx in range(self.num_dispatchers):
            self.dispatch_queues.append(queue.Queue())
            dispatcher = threading.Thread(target=self.dispatcher_target, args=(dispatch_idx,))
            self.dispatchers.append(dispatcher)
            dispatcher.start()

        self.dispatch_idx = 0

    def dispatcher_target(self, dispatch_idx: int):
        if self.verbose:
            logging.info(f"Starting dispatcher #{dispatch_idx} to model {self.triton_model_name}")
        client = self.clients[dispatch_idx]
        dispatch_queue = self.dispatch_queues[dispatch_idx]
        grpc_callback = self.handle_queries_callback
        if self.llm_config.streaming:
            client.start_stream(callback=grpc_callback)

        while True:
            sample = dispatch_queue.get()
            if sample is None:
                dispatch_queue.task_done()
                break
            sample_input_ids, sample_input_lens = self._ds.get_input(sample.index)
            inputs = get_gptj_triton_inputs(tensor_input_ids=sample_input_ids,
                                            tensor_input_len=sample_input_lens,
                                            llm_config=self.llm_config,
                                            protocol="grpc")
            outputs = []
            for output in ["output_ids", "sequence_length"]:
                outputs.append(grpcclient.InferRequestedOutput(output))

            if self.log_stats:
                self._update_disp_sample_stats(sample.id, sample_input_lens)
            if self.llm_config.streaming:
                client.async_stream_infer(model_name=self.triton_model_name, inputs=inputs, request_id=str(sample.id), outputs=outputs)
            else:
                client.async_infer(model_name=self.triton_model_name, inputs=inputs, callback=grpc_callback, request_id=str(sample.id), outputs=outputs)

            self.num_queries_dispatched += 1
            dispatch_queue.task_done()
        if self.verbose:
            logging.info(f"Stopping dispatcher #{dispatch_idx} to model {self.triton_model_name}")

        if self.llm_config.streaming:
            client.stop_stream(cancel_requests=False)

        dispatch_queue.join()
        client.close()

    def _dump_stats(self):
        pass

    def _update_disp_sample_stats(self, sample_id, isl):
        pass

    def dispatch_query_samples(self, query_samples):
        with self.dispatching_lock:
            if not type(query_samples) is list:
                query_samples = [query_samples]
            for query_sample in query_samples:
                self.dispatch_queues[self.dispatch_idx].put(query_sample)
                self.num_total_queries_dispatched += 1
                self.dispatch_idx += 1
                self.dispatch_idx %= self.num_dispatchers
                if self.verbose:
                    if self.num_total_queries_dispatched % 1000 == 0:
                        logging.info(f"Dispatched {self.num_total_queries_dispatched} samples to {self.triton_model_name}")

    def wait_for_server_readiness(self, is_server_ready: Callable[[], bool], poll_interval: int = 1):
        server_ready = False
        logging.info(f"waiting for server/{self.triton_model_name} to be ready")
        while not server_ready:
            try:
                server_ready = is_server_ready()
                if server_ready:
                    break
                else:
                    time.sleep(poll_interval)
            except (ConnectionRefusedError, InferenceServerException):
                time.sleep(poll_interval)

    def notify_dispatch_done(self):
        logging.info(f"Closing frontend to {self.triton_model_name}")
        with self.dispatching_lock:
            for queue in self.dispatch_queues:
                queue.put(None)
            for dispatcher in self.dispatchers:
                dispatcher.join()
        self.report_loadgen_queue.put(None)

    @abstractmethod
    def handle_queries_callback(self, result, error):
        pass


class TritonSutGrpcFrontend(ITritonSutFrontend):
    def __init__(self,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

    def handle_queries_callback(self, result, error):
        assert error is None, "Inference not successful"
        self.queries_responded += 1

        sample_id = int(result.get_response().id)
        output_ids_tensor = result.as_numpy("output_ids")
        output_len_tensor = result.as_numpy("sequence_length")
        n_tokens = output_len_tensor[0, 0]
        output_ids_tensor = output_ids_tensor[:, 0, :n_tokens].reshape(1, -1)  # first beam

        while n_tokens <= 1:
            output_ids_tensor = np.append(output_ids_tensor, [[self.llm_config.end_token_id]], axis=1)
            n_tokens = output_ids_tensor.shape[1]

        if self.report_loadgen_queue is not None:
            self.report_loadgen_queue.put([False, sample_id, output_ids_tensor])
        else:
            curr_qsr = lg.QuerySampleResponse(sample_id, output_ids_tensor.ctypes.data, byte_size, n_tokens)
            lg.QuerySamplesComplete([curr_qsr])
        if self.verbose:
            if self.queries_responded % 1000 == 0:
                logging.info(f"Model {self.triton_model_name} completed inference of {self.queries_responded} samples")


class TritonSutGrpcStreamFrontend(ITritonSutFrontend):
    def __init__(self,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accumulated_output_tokens = {}
        self.num_first_tokens_recvd = 0
        self.num_tokens_completed = 0

        self.recv_first_token_queue = queue.Queue()
        self.first_token_gatherer_thread = threading.Thread(target=self.first_token_gatherer)
        self.first_token_gatherer_thread.start()

        self.recv_interm_token_queue = queue.Queue()
        self.interm_token_gatherer_thread = threading.Thread(target=self.intermediate_token_gatherer)
        self.interm_token_gatherer_thread.start()

        self.token_recv_stats = [['sample_id', 'tok_type', 'num_dispatched', 'gpu_util', 'out_len', 'timestamp'], ]
        self.token_recv_stats.extend([[0] * 6] * 100_000_000)  # reserve space for 100M tokens
        self.token_recv_stats_counter = 0
        self.dispatched_sample_stats = [['sample_id', 'timestamp', 'isl'], ]

        self.token_gatherer_lock = [threading.Lock(), threading.Lock()]  # one for first tokens, one for final tokens

        if importlib.util.find_spec('nvidia_smi') is not None:
            self.nvsmi_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        else:
            self.nvsmi_handle = None

    def handle_queries_callback(self, result, error):
        assert error is None
        interm_tokens = self.accumulated_output_tokens
        response = result.get_response()
        sample_id = int(response.id)
        is_first_token = sample_id not in interm_tokens

        if is_first_token:
            self.recv_first_token_queue.put(result)
        else:
            self.recv_interm_token_queue.put(result)

    def first_token_gatherer(self):
        recv_queue = self.recv_first_token_queue
        while True:
            result = recv_queue.get()
            if result is None:
                recv_queue.task_done()
                break
            self._gather_first_token(result)
            recv_queue.task_done()
        recv_queue.join()

    def intermediate_token_gatherer(self):
        while True:
            result = self.recv_interm_token_queue.get()

            if result is None:
                if self.report_loadgen_queue is not None:
                    self.report_loadgen_queue.put(None)
                self.recv_interm_token_queue.task_done()
                break
            self._gather_interm_token(result)
            self.recv_interm_token_queue.task_done()
        self.recv_interm_token_queue.join()

    def _gather_first_token(self, result):
        interm_tokens = self.accumulated_output_tokens
        response = result.get_response()
        sample_id = int(response.id)
        output_ids_tensor = result.as_numpy("output_ids")
        output_len_tensor = result.as_numpy("sequence_length")
        n_tokens = output_len_tensor[0, 0]
        recvd_token = output_ids_tensor[0, 0]
        assert n_tokens == 1
        assert recvd_token != self.llm_config.end_token_id
        is_final = response.parameters.get("triton_final_response").bool_param
        is_final = is_final or output_ids_tensor[0, 0] == self.llm_config.end_token_id

        if self.report_loadgen_queue is None:
            curr_qsr = lg.QuerySampleResponse(sample_id, output_ids_tensor.ctypes.data, 4, 1)
            lg.FirstTokenComplete([curr_qsr])

            if is_final:
                eos_tensor = np.array([[self.llm_config.end_token_id]], dtype=np.int32)
                curr_qsr = lg.QuerySampleResponse(sample_id, eos_tensor.ctypes.data, 4, 1)
                lg.QuerySamplesComplete([curr_qsr])
        else:
            self.report_loadgen_queue.put([True, sample_id, output_ids_tensor])
            if is_final:
                eos_tensor = np.array([[self.llm_config.end_token_id]], dtype=np.int32)
                self.report_loadgen_queue.put([False, sample_id, eos_tensor])

        interm_tokens[sample_id] = [recvd_token]
        if self.log_stats:
            self._update_token_stats(sample_id=sample_id, tok_type='1', num_dispatched=self.num_queries_dispatched)

        with self.token_gatherer_lock[1]:
            self.num_first_tokens_recvd += 1

        if self.verbose:
            if self.num_first_tokens_recvd % 1000 == 0:
                logging.info(f"Model {self.triton_model_name} received {self.num_first_tokens_recvd} first tokens, in-flight reqs: {len(interm_tokens)}")

    def _gather_interm_token(self, result):
        output_len_tensor = result.as_numpy("sequence_length")
        n_tokens = output_len_tensor[0, 0]
        output_ids_tensor = result.as_numpy("output_ids")
        output_ids_tensor = output_ids_tensor[:, 0, :n_tokens].reshape(1, -1)  # first beam

        response = result.get_response()
        sample_id = int(response.id)
        is_final = response.parameters.get("triton_final_response").bool_param
        is_final = is_final or n_tokens == 0
        if n_tokens == 1:
            is_final = is_final or output_ids_tensor[0, 0] == self.llm_config.end_token_id
        interm_tokens = self.accumulated_output_tokens

        assert n_tokens <= 1, "In streaming mode, we expect <= 1 tokens per response, got: {}".format(n_tokens)
        assert output_ids_tensor.shape[1] == n_tokens, "mismatching output_ids and sequence_length"
        assert sample_id in interm_tokens

        if is_final:
            output_ids = interm_tokens.pop(sample_id, None)

            if n_tokens == 1:
                output_ids.append(output_ids_tensor[0, 0])

            output_ids.append(self.llm_config.end_token_id)

            seq_len = len(output_ids)
            output_ids_tensor = np.asarray(output_ids, dtype=np.int32).reshape(1, -1)

            if self.report_loadgen_queue is None:
                curr_qsr = lg.QuerySampleResponse(sample_id, output_ids_tensor.ctypes.data, 4 * seq_len, seq_len)
                lg.QuerySamplesComplete([curr_qsr])
            else:
                self.report_loadgen_queue.put([False, sample_id, output_ids_tensor])

            if self.log_stats:
                self._update_token_stats(sample_id=sample_id, tok_type='C', num_dispatched=self.num_queries_dispatched, out_len=seq_len)

            with self.token_gatherer_lock[0]:
                self.queries_responded += 1
                self.num_tokens_completed += seq_len

            if self.verbose:
                if self.queries_responded % 1000 == 0:
                    logging.info(f"Model {self.triton_model_name} completed inference of {self.queries_responded} samples, {self.num_tokens_completed} tokens")

        else:  # intermediate token
            assert n_tokens == 1
            recvd_token = output_ids_tensor[0, 0]
            interm_tokens[sample_id].append(recvd_token)
            if self.log_stats:
                self._update_token_stats(sample_id=sample_id, tok_type='I', num_dispatched=self.num_queries_dispatched)

    def _update_token_stats(self, sample_id, tok_type, num_dispatched, **kwargs):
        util = -1
        out_len = -1
        if 'out_len' in kwargs:
            out_len = kwargs['out_len']
        if self.nvsmi_handle is not None:
            util = nvidia_smi.nvmlDeviceGetUtilizationRates(self.nvsmi_handle).gpu
        l = [sample_id, tok_type, num_dispatched, util, out_len, time.time()]
        count = self.token_recv_stats_counter
        self.token_recv_stats[count + 1] = l
        self.token_recv_stats_counter += 1

    def _update_disp_sample_stats(self, sample_id, isl):
        self.dispatched_sample_stats.append([str(sample_id), time.time(), int(isl[0, 0]), ])

    def _dump_stats(self):
        self.token_recv_stats = self.token_recv_stats[:(self.token_recv_stats_counter + 1)]
        with open('token_recv_stats.json', 'w') as f:
            json.dump(self.token_recv_stats, f)
        with open('sent_sample_stats.json', 'w') as f:
            json.dump(self.dispatched_sample_stats, f)

    def notify_dispatch_done(self):
        super().notify_dispatch_done()
        with self.dispatching_lock:
            while self.recv_first_token_queue.unfinished_tasks > 0:
                logging.info("{self.triton_model_name}: waiting to consume all tokens in recv_first_token_queue")
                time.sleep(1)
            self.recv_first_token_queue.put(None)
            self.first_token_gatherer_thread.join()

            while self.recv_interm_token_queue.unfinished_tasks > 0:
                logging.info("{self.triton_model_name}: waiting to consume all tokens in intermediate token queue")
                time.sleep(1)
            self.recv_interm_token_queue.put(None)
            self.interm_token_gatherer_thread.join()
            if self.log_stats:
                self._dump_stats()
