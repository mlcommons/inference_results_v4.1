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

from code.harness.harness_triton_llm.dataset import LlmDataset
from code.harness.harness_triton_llm.frontend import TritonSutGrpcFrontend, TritonSutGrpcStreamFrontend
from code.harness.harness_triton_llm.backend import TritonSutBackend
from code.harness.harness_triton_llm.utils import LlmConfig, get_llm_gen_config

import argparse
from pathlib import Path
import mlperf_loadgen as lg
import traceback
import threading
import queue
import logging
import json
import multiprocessing as mp
from functools import partial

G_DEFAULT_PORTS = {'http': 8000, 'grpc': 8001, 'metrics': 8002}
G_FRONTEND_DISPATCH_IDX = 0
G_NUM_DISPATCHED = 0

scenario_map = {
    "Offline": lg.TestScenario.Offline,
    "SingleStream": lg.TestScenario.SingleStream,
    "Server": lg.TestScenario.Server,
}

test_mode_map = {
    "PerformanceOnly": lg.TestMode.PerformanceOnly,
    "AccuracyOnly": lg.TestMode.AccuracyOnly,
    "SubmissionRun": lg.TestMode.SubmissionRun,
}

log_mode_map = {
    "AsyncPoll": lg.LoggingMode.AsyncPoll,
    "EndOfTestOnly": lg.LoggingMode.EndOfTestOnly,
    "Synchronous": lg.LoggingMode.Synchronous,
}


def dummy_flush():
    pass


def flush_queries(frontend_input_queues):
    for input_queue in frontend_input_queues:
        input_queue.put(None)


def dispatch_queries_from_loadgen(frontend_input_queues, query_samples):
    global G_FRONTEND_DISPATCH_IDX
    global G_NUM_DISPATCHED
    num_frontends = len(frontend_input_queues)
    for query_sample in query_samples:
        frontend_input_queues[G_FRONTEND_DISPATCH_IDX].put(query_sample, block=True)
        G_FRONTEND_DISPATCH_IDX += 1
        G_FRONTEND_DISPATCH_IDX %= num_frontends
        G_NUM_DISPATCHED += 1
        if G_NUM_DISPATCHED % 1000 == 0:
            logging.info(f"Sent {G_NUM_DISPATCHED} samples")


def frontend_process(frontend_class: type,
                     input_queue: mp.Queue,
                     triton_model_name: str,
                     grpc_url: str,
                     dataset: LlmDataset,
                     llm_config: LlmConfig,
                     args: argparse.Namespace,
                     frontend_ready_queue: mp.Queue,
                     output_qsr_queue: mp.Queue):
    logging.info(f"Initializing frontend for {triton_model_name}")
    frontend = frontend_class(dataset=dataset,
                              llm_config=llm_config,
                              verbose=args.verbose_frontend,
                              triton_model_name=triton_model_name,
                              url=grpc_url,
                              num_clients_per_gpu=args.num_clients_per_gpu,
                              report_loadgen_queue=output_qsr_queue)
    frontend_ready_queue.put(None)

    while True:
        query_sample = input_queue.get()
        if query_sample is None:
            break
        frontend.dispatch_query_samples(query_sample)
    frontend.notify_dispatch_done()


def send_results_to_loadgen(output_queue: mp.Queue, verbose_frontend: bool):
    num_first_toks_recvd = 0
    num_samples_completed = 0
    while True:
        result = output_queue.get()
        if result is None:
            break
        is_first, sample_id, output_ids = result
        if is_first:
            qsr = lg.QuerySampleResponse(sample_id, output_ids.ctypes.data, 4, 1)
            lg.FirstTokenComplete([qsr])
            num_first_toks_recvd += 1
            if verbose_frontend:
                if num_first_toks_recvd % 1000 == 0:
                    logging.info(f"Received {num_first_toks_recvd} first tokens.")
        else:
            seq_len = output_ids.shape[-1]
            qsr = lg.QuerySampleResponse(sample_id, output_ids.ctypes.data, 4 * seq_len, seq_len)
            lg.QuerySamplesComplete([qsr])
            num_samples_completed += 1
            if verbose_frontend:
                if num_samples_completed % 1000 == 0:
                    logging.info(f"Completed {num_samples_completed} samples")


def parse_args():
    parser = argparse.ArgumentParser()

    # Test args
    parser.add_argument("--scenario", choices=["Offline", "Server"], default="Offline")
    parser.add_argument("--test_mode", choices=["PerformanceOnly", "AccuracyOnly", "SubmissionRun"], default="PerformanceOnly")
    parser.add_argument("--model", choices=["gptj", "llama2-70b"], default="gptj")
    parser.add_argument("--llm_gen_config_path", help="Path to generation_config.json")
    parser.add_argument("--num_gpus", help="Number of GPUs to use for tritonserver. The harness spawns identical triton models - one per GPU", default=0, type=int)

    # QSL args
    parser.add_argument("--tensor_path", type=str, help="(comma separated) path(s) to the dataset tensor file")
    parser.add_argument("--dispatcher_type", type=str, choices=["sequential", "mlperf"], default="mlperf", help="The dispatching behavior of queries.")

    # Config args
    parser.add_argument("--performance_sample_count", type=int, default=5000, help="Number of samples to run benchmark on")
    parser.add_argument("--mlperf_conf_path", help="Path to mlperf.conf", default="build/loadgen-configs/DGX-H100_H100-SXM-80GBx1_TRT/gptj-99/Offline/mlperf.conf")
    parser.add_argument("--user_conf_path", help="Path to user.conf", default="build/loadgen-configs/DGX-H100_H100-SXM-80GBx1_TRT/gptj-99/Offline/user.conf")

    # Log args
    parser.add_argument("--log_mode", type=str, default="AsyncPoll", help="Logging mode for Loadgen")
    parser.add_argument("--log_mode_async_poll_interval_ms", type=int, default=1000, help="Specify the poll interval for asynchrounous logging")
    parser.add_argument("--logfile_outdir", type=str, default='/work/build/logs/triton', help="Specify the existing output directory for the LoadGen logs")
    parser.add_argument("--logfile_prefix", type=str, default='triton-grpc', help="Specify the filename prefix for the LoadGen log files")
    parser.add_argument("--logfile_suffix", type=str, default='', help="Specify the filename suffix for the LoadGen log files")

    # Triton control knobs
    parser.add_argument("--skip_server_spawn", action="store_true", help="Skip starting a tritonserver process")
    parser.add_argument("--verbose_frontend", action="store_true", help="Make triton frontend verbose, this enables logging of stats")
    parser.add_argument("--protocol", type=str, default="grpc", help="Protocol to use for triton client-server communication")
    parser.add_argument("--model_repo", type=str, default="/work/build/triton_model_repo", help="Model repo path for tritonserver")
    parser.add_argument("--num_clients_per_gpu", type=int, default=1, help="Number of triton clients per GPU. Total number of clients are (num_gpus * num_clients_per_gpu)")

    args, _ = parser.parse_known_args()
    assert args.num_gpus > 0, "num GPUs must be a positive integer"
    return args


if __name__ == "__main__":
    mp.set_start_method('spawn')
    args = parse_args()
    llm_config = get_llm_gen_config(args.model, args.scenario, args.llm_gen_config_path)

    # Initialize settings
    test_settings = lg.TestSettings()
    test_settings.scenario = scenario_map[args.scenario]
    test_settings.mode = test_mode_map[args.test_mode]

    # Load config
    test_settings.FromConfig(args.mlperf_conf_path, args.model, args.scenario)
    test_settings.FromConfig(args.user_conf_path, args.model, args.scenario)
    test_settings.server_coalesce_queries = True

    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = args.logfile_outdir
    log_output_settings.prefix = args.logfile_prefix
    log_output_settings.suffix = args.logfile_suffix
    log_output_settings.copy_summary_to_stdout = True
    Path(args.logfile_outdir).mkdir(parents=True, exist_ok=True)

    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.log_mode = log_mode_map[args.log_mode]
    log_settings.log_mode_async_poll_interval_ms = args.log_mode_async_poll_interval_ms

    # Initialize QSL
    input_ids, input_lens = args.tensor_path.split(',')
    dataset = LlmDataset(path_input_ids=input_ids, path_input_lens=input_lens)
    qsl = lg.ConstructQSL(dataset.get_size(), args.performance_sample_count, dataset.load_samples_to_ram, dataset.unload_samples_from_ram)

    frontend_class = TritonSutGrpcFrontend
    if llm_config.streaming:
        frontend_class = TritonSutGrpcStreamFrontend

    model_name_prefix = llm_config.model_name.lower() + '-' + args.scenario.lower()

    http_port = G_DEFAULT_PORTS['http']
    grpc_port = G_DEFAULT_PORTS['grpc']
    metrics_port = G_DEFAULT_PORTS['metrics']
    if not args.skip_server_spawn:
        backend = TritonSutBackend(model_name_prefix=model_name_prefix, model_repo=args.model_repo, num_gpus=args.num_gpus)
        grpc_port = backend.get_grpc_port()
        http_port = backend.get_http_port()
        metrics_port = backend.get_metrics_port()

    grpc_url = f"0.0.0.0:{grpc_port}"

    children_processes = []
    children_input_queues = []
    children_ready_queues = []
    children_output_queues = []
    for gpu_idx in range(args.num_gpus):
        # spawn a child process that will hold a frontend.
        # This frontend will be responsible for sending queries to GPU #{gpu_idx}
        triton_model_name = model_name_prefix + '-' + str(gpu_idx)
        frontend_input_queue = mp.Queue(maxsize=1000)
        frontend_output_queue = mp.Queue(maxsize=1000)
        frontend_ready_queue = mp.Queue()

        child = mp.Process(target=frontend_process, args=(frontend_class, frontend_input_queue,
                                                          triton_model_name, grpc_url, dataset,
                                                          llm_config, args, frontend_ready_queue,
                                                          frontend_output_queue))
        children_processes.append(child)
        children_input_queues.append(frontend_input_queue)
        children_output_queues.append(frontend_output_queue)
        children_ready_queues.append(frontend_ready_queue)
        child.start()

    qsr_consumers = []
    for output_queue in children_output_queues:
        qsr_consumer = threading.Thread(target=send_results_to_loadgen, args=(output_queue, args.verbose_frontend,))
        qsr_consumer.start()
        qsr_consumers.append(qsr_consumer)

    for queue in children_ready_queues:
        _ = queue.get()
        queue.close()

    sut = lg.ConstructSUT(partial(dispatch_queries_from_loadgen, children_input_queues),
                          dummy_flush)
    logging.info("Initialized TritonSUT. Starting benchmark run")

    # Start test
    lg.StartTestWithLogSettings(sut, qsl, test_settings, log_settings)
    logging.info("Benchmark run complete")

    for input_queue in children_input_queues:
        input_queue.put(None)

    for child in children_processes:
        child.join()

    for qsr_consumer in qsr_consumers:
        qsr_consumer.join()

    # Destroying SuT, Qsl
    lg.DestroySUT(sut)
    logging.info("Destroyed SUT")

    lg.DestroyQSL(qsl)
    logging.info("Destroyed QSL")
