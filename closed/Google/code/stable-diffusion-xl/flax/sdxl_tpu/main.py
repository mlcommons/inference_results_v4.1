"""
mlperf inference benchmarking tool
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import concurrent.futures
import argparse
import array
import json
import logging
import os
import sys
import time
from collections import deque
import jax.numpy as jnp
from typing import List

import mlperf_loadgen as lg
import numpy as np

import dataset
import coco

import requests
from typing import List
from backend_jax import StableDiffusion

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

from threading import Thread, local

NANO_SEC = 1e9
MILLI_SEC = 1000

SUPPORTED_DATASETS = {
    "coco-1024": (
        coco.Coco,
        dataset.preprocess,
        coco.PostProcessCoco(),
        {"image_size": [3, 1024, 1024]},
    )
}


SUPPORTED_PROFILES = {
    "defaults": {
        "dataset": "coco-1024",
        "backend": "jax",
        "model-name": "stable-diffusion-xl",
    }
}

SCENARIO_MAP = {
    "Server": lg.TestScenario.Server,
    "Offline": lg.TestScenario.Offline,
}

thread_local = local()

def get_session():
    if not hasattr(thread_local,'session'):
        thread_local.session = requests.Session()
    return thread_local.session

def construct_request(prompts):
    d = {'instances': []}
    for p in prompts:
        d['instances'].append({'prompt': p, 'query_id':["1"]})
    return d


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=SUPPORTED_DATASETS.keys(), help="dataset")
    parser.add_argument("--dataset-path", required=False,
                        help="path to the dataset", default="coco2014")
    parser.add_argument("--config", required=True, help="path to the dataset")
    parser.add_argument("--latents", required=True, help="path to the latents")

    parser.add_argument(
        "--profile", choices=SUPPORTED_PROFILES.keys(), help="standard profiles"
    )
    parser.add_argument(
        "--scenario",
        default="Server",
        help="mlperf benchmark scenario, one of " +
            str(list(SCENARIO_MAP.keys())),
    )
    parser.add_argument(
        "--max-batchsize",
        type=int,
        default=64,
        help="max batch size in a single inference",
    )

    parser.add_argument(
        "--threshold-time",
        type=int,
        default=8,
        help="max batch size in a single inference",
    )

    parser.add_argument(
        "--threshold-queue-length",
        type=int,
        default=4,
        help="max batch size in a single inference",
    )

    parser.add_argument("--threads", default=8, type=int, help="threads")
    parser.add_argument("--accuracy", action="store_true",
                        help="enable accuracy pass")
    parser.add_argument(
        "--find-peak-performance",
        action="store_true",
        help="enable finding peak performance pass",
    )
    parser.add_argument("--backend", help="Name of the backend")
    parser.add_argument("--model-name", help="Name of the model")
    parser.add_argument("--output", default="output", help="test results")
    parser.add_argument("--qps", type=int, help="target qps")
    parser.add_argument("--model-path", help="Path to model weights")
    parser.add_argument(
        "--samples-per-query",
        default=8,
        type=int,
        help="mlperf multi-stream samples per query",
    )

    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["fp32"],
        help="dtype of the model",
    )

    parser.add_argument(
        "--latent-framework",
        default="numpy",
        choices=["numpy"],
        help="framework to load the latents",
    )

    # file to use mlperf rules compliant parameters
    parser.add_argument(
        "--mlperf_conf", default="mlperf.conf", help="mlperf rules config"
    )
    # file for user LoadGen settings such as target QPS
    parser.add_argument(
        "--user_conf",
        default="user.conf",
        help="user config for user LoadGen settings such as target QPS",
    )
    # file for LoadGen audit settings
    parser.add_argument(
        "--audit_conf", default="audit.config", help="config for LoadGen audit settings"
    )
    # arguments to save images
    # pass this argument for official submission
    # parser.add_argument("--output-images", action="store_true", help="Store a subset of the generated images")
    # do not modify this argument for official submission
    parser.add_argument("--ids-path", help="Path to caption ids",
                        default="tools/sample_ids.txt")

    # below will override mlperf rules compliant settings - don't use for official submission
    parser.add_argument("--time", type=int, help="time to scan in seconds")
    parser.add_argument("--count", type=int, help="dataset items to use")
    parser.add_argument("--debug", action="store_true", help="debug")
    parser.add_argument(
        "--performance-sample-count", type=int, help="performance sample count", default=5000
    )
    parser.add_argument(
        "--max-latency", type=float, help="mlperf max latency in pct tile"
    )

    args, unknown = parser.parse_known_args()

    # don't use defaults in argparser. Instead we default to a dict, override that with a profile
    # and take this as default unless command line give
    defaults = SUPPORTED_PROFILES["defaults"]

    if args.profile:
        profile = SUPPORTED_PROFILES[args.profile]
        defaults.update(profile)
    for k, v in defaults.items():
        kc = k.replace("-", "_")
        if getattr(args, kc) is None:
            setattr(args, kc, v)

    if args.scenario not in SCENARIO_MAP:
        parser.error("valid scanarios:" + str(list(SCENARIO_MAP.keys())))
    return args

class Item:
    """An item that we queue for processing by the thread pool."""

    def __init__(self, query_id, content_id, inputs, img=None):
        self.query_id = query_id
        self.content_id = content_id
        self.img = img
        self.inputs = inputs
        self.start = time.time()

class ThreadedSDClientOffline:
  """Holds a thread pool and a sax client for LM inference."""

  _thread_pool: concurrent.futures.ThreadPoolExecutor
  _dataset: dataset.Dataset
  _futures = List[concurrent.futures.Future]

  def __init__(
      self,
      args,
      dataset,
      log_interval: int = 1000,
  ):
    log.info(f"Initiating {self.__class__.__name__} ...")
    self._log_interval = log_interval
    self._thread_pool = concurrent.futures.ThreadPoolExecutor(args.threads)
    self._futures = []
    self._resp_cnt = 0
    self.ds = dataset
    self.args = args
    self.model = StableDiffusion(dataset_path=args.dataset_path, config=self.args.config, latents_path=self.args.latents)
    _images = self.model.predict(["warming up"]) # warm up query
    log.info(f"Warmed up images length: {len(_images)}")

  def _log_resp_cnt(self):
    self._resp_cnt += 1
    if self._resp_cnt % self._log_interval == 0:
      log.info("Completed %d queries", self._resp_cnt)

  def process_single_sample_async(self, prompts, content_ids, query_ids):
    """Executes a single query and marks responses complete asynchronously.

    Args:
      query_sample: Single prompt
      warmup: Indicates that this is a warmup request.
    """
    future = self._thread_pool.submit(
        self._process_sample, prompts, content_ids, query_ids
    )
    self._futures.append(future)
  
  def flush(self):
    log.info("Flushing queries")
    future_count = 0
    num_logged = 0
    for future in concurrent.futures.as_completed(self._futures):
        try:
            content_id, res_np_array = future.result() # response_array_list, bi_0_list, bi_1_list = future.result()
            self.ds.content_ids.extend(content_id)
            self.ds.results.extend(res_np_array)
            num_logged += len(res_np_array)
        except Exception as exc:
            print("generated an exception:")
            print(repr(exc))
        future_count += 1
    log.info("Completed All Futures")
    log.info(f"Successfully predicted count: {num_logged}")
    self._futures = []

  def _process_sample(self, prompts, content_ids, query_ids):
    """Processes a single sample."""
    try:
        log.info(f"Sampling for:  {self._resp_cnt}")
        predictions = self.model.predict(prompts)

        result_arr = []
        log.info("Json Parsed")
        response_array_refs = []

        response = []
        for i, im in enumerate(predictions):
            # res_np = np.asarray(im)
            processed_result = np.squeeze(im)
            result_arr.append(processed_result)
            response_array = array.array("B", np.array(processed_result, np.uint8).tobytes())
            bi = response_array.buffer_info()
            response_array_refs.append(response_array)
            log.info("Image Postprocess done")
            response.append(lg.QuerySampleResponse(query_ids[i], bi[0], bi[1]))
            log.info(f"QuerySampleResponse done for query: {i}")
        lg.QuerySamplesComplete(response)


        return content_ids, result_arr
    except Exception as e:
        print(repr(e))


class ThreadedSDClientServer:
  """Holds a thread pool and a sax client for LM inference."""

  _thread_pool: concurrent.futures.ThreadPoolExecutor
  _dataset: dataset.Dataset
  _futures = List[concurrent.futures.Future]
  def __init__(
      self,
      args,
      dataset,
      log_interval: int = 1000,
  ):
    log.info(f"Initiating {self.__class__.__name__} ...")
    self._log_interval = log_interval
    self._thread_pool = concurrent.futures.ThreadPoolExecutor(args.threads)
    self._futures = []
    self._resp_cnt = 0
    self.ds = dataset
    self.st = StableDiffusion(dataset_path=args.dataset_path, config=args.config, latents_path=args.latents)
    print("starting warmup")
    for i in range(2):
        self.st.predict(["warming up"])
    print("done warming up")

  def _log_resp_cnt(self, l):
    self._resp_cnt += l
    if self._resp_cnt % self._log_interval == 0:
      log.info("Completed %d queries", self._resp_cnt)

  def process_single_sample_async(self, prompts, content_ids, query_ids):
    """Executes a single query and marks responses complete asynchronously.

    Args:
      query_sample: Single prompt
      warmup: Indicates that this is a warmup request.
    """
    future = self._thread_pool.submit(
        self._process_sample_server, prompts, content_ids, query_ids
    )
    self._futures.append(future)

  def flush(self):
    log.info("Flushing queries")
    self._futures = []
      
      
  
  def _process_sample_server(self, prompts, content_ids, query_ids):
    """Processes a single sample."""
    try:
        log.info(f"Sampling for:  {self._resp_cnt}")
        start_time = time.time()
        images = self.st.predict([p[0] for p in prompts])
        print("Images shape:", images.shape)
        print(f"Response time: {time.time() - start_time}")
        result_arr = []
        response_array_refs = []
        for i in range(len(prompts)):
            res_np = images[i]
            response = []
            processed_result = np.squeeze(res_np)
            result_arr.append(processed_result)
            response_array = array.array(
                "B", np.array(processed_result, np.uint8).tobytes()
            )
            bi = response_array.buffer_info()
            response_array_refs.append(response_array)
            response.append(lg.QuerySampleResponse(query_ids[i][0], bi[0], bi[1]))
            log.info("QuerySampleResponse done ")
            lg.QuerySamplesComplete(response)
            log.info(f"Completed response for content_id: {content_ids[i]}, query_id: {query_ids[i]}, prompt: {prompts[i]}")
            
            self.ds.content_ids.extend(content_ids[i])
            self.ds.results.append(res_np)
        self._log_resp_cnt(len(prompts))

        return content_ids, result_arr
    except Exception as e:
        print(repr(e))
        


class QueueRunner:
    def __init__(self, args, ds, post_proc=None, scenario="Server"):
        self.ds = ds
        self.post_process = post_proc
        self.max_batchsize = args.max_batchsize
        self.result_dict = {}
        self.scenario = scenario
        threaded_client = ThreadedSDClientServer if scenario == "Server" else ThreadedSDClientOffline
        self._client = threaded_client(args, self.post_process)
        self.query_number = 0
        self.queue = deque()
        self.th_length = args.threshold_queue_length #4
        self.t_thresh = args.threshold_time #8
        self.t0 = time.time()
        self.background_thread = None
    
    def run_background_queue_clearer(self):
        while True:
            items_to_process = []
            if len(self.queue) == 0:
                return
            if len(self.queue) >= self.th_length:
                for i in range(self.th_length):
                    items_to_process.append(self.queue.popleft())
            elif time.time() - self.t0 >= self.t_thresh:
                for i in range(len(self.queue)):
                    items_to_process.append(self.queue.popleft())
                

            if items_to_process:
                self.t0 = time.time()
                self._client._process_sample_server([t[0] for t in items_to_process] , 
                                                            [t[1] for t in items_to_process] , 
                                                            [t[2] for t in items_to_process])
    def issue_queries(self, query_samples):
        if self.scenario == "Server":
            num_query_samples = len(query_samples)
            log.info(f"Issuing {num_query_samples} queries. ")  
            content_ids = [q.index for q in query_samples]
            query_ids = [q.id for q in query_samples]
            data, label = self.ds.get_samples(content_ids)
            captions = [d["input_captions"] for d in data]
            if not self.queue:
                print("queue empty")
                if self.background_thread is not None: self.background_thread.join()
                self.queue.append((captions, content_ids, query_ids))
                self.background_thread = Thread(target=self.run_background_queue_clearer, args=())
                self.background_thread.daemon = True
                self.background_thread.start()
            else:
                print("queue size:", len(self.queue))
                self.queue.append((captions, content_ids, query_ids))
            
            
            self.query_number += 1
            print("Issued Queries: ", self.query_number)
        else:
            num_query_samples = len(query_samples)
            log.info(f"Issuing {num_query_samples} queries. ")  
            bs = self.max_batchsize
            for i in range(0, num_query_samples, bs):
                ie = i + bs
                print(">>> Putting to queue: i, ie, bs", i, ie, bs)
                content_ids = [q.index for q in query_samples][i:ie]
                query_ids = [q.id for q in query_samples][i:ie]
                data, label = self.ds.get_samples(content_ids)
                captions = [d["input_captions"] for d in data]
                print(len(content_ids), len(query_ids), len(data))       
                self._client.process_single_sample_async(captions, content_ids, query_ids)
            print("done issuing queries")
        
    def flush_queries(self):
        """Flush queries."""
        log.info("Loadgen has completed issuing queries... ")
        self._client.flush()
        if self.background_thread is not None:
            self.background_thread.join()

        print("flushed queries ")
    
    def finish(self):
        pass
    

def main():
    args = get_args()
    log.info(args)

    if args.dtype == "bf16":
        dtype = jnp.bfloat16
    else:
        dtype = jnp.float32

    # --count applies to accuracy mode only and can be used to limit the number of images
    # for testing.
    count_override = False
    count = args.count
    if count:
        count_override = True

    # dataset to use
    dataset_class, pre_proc, post_proc, kwargs = SUPPORTED_DATASETS[args.dataset]
    ds = dataset_class(
        data_path=args.dataset_path,
        name=args.dataset,
        pre_process=pre_proc,
        count=count,
        threads=args.threads,
        latent_dtype="",
        pipeline=None,
        latent_framework=args.latent_framework,
        **kwargs,
    )
    final_results = {
        "runtime": "SDXL-Base",
        "version": "1.0",
        "time": int(time.time()),
        "args": vars(args),
        "cmdline": str(args),
    }

    mlperf_conf = os.path.abspath(args.mlperf_conf)
    if not os.path.exists(mlperf_conf):
        log.error("{} not found".format(mlperf_conf))
        sys.exit(1)

    user_conf = os.path.abspath(args.user_conf)
    if not os.path.exists(user_conf):
        log.error("{} not found".format(user_conf))
        sys.exit(1)

    audit_config = os.path.abspath(args.audit_conf)
    
    if args.accuracy:
        ids_path = os.path.abspath(args.ids_path)
        with open(ids_path) as f:
            saved_images_ids = [int(_) for _ in f.readlines()]

    if args.output:
        output_dir = os.path.abspath(args.output)
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)

    #
    # make one pass over the dataset to validate accuracy
    #
    count = ds.get_item_count()

    scenario = SCENARIO_MAP[args.scenario]
    runner_map = {
        lg.TestScenario.Server: QueueRunner,
        lg.TestScenario.Offline: QueueRunner,
    }
    runner = runner_map[scenario](
        args, ds, post_proc=post_proc, scenario=args.scenario
    )

    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = output_dir
    log_output_settings.copy_summary_to_stdout = False
    log_settings = lg.LogSettings()
    log_settings.enable_trace = args.debug
    log_settings.log_output = log_output_settings

    settings = lg.TestSettings()
    settings.FromConfig(mlperf_conf, args.model_name, args.scenario)
    settings.FromConfig(user_conf, args.model_name, args.scenario)
    settings.scenario = scenario
    settings.mode = lg.TestMode.PerformanceOnly
    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    if args.find_peak_performance:
        settings.mode = lg.TestMode.FindPeakPerformance

    if args.time:
        # override the time we want to run
        settings.min_duration_ms = args.time * MILLI_SEC
        settings.max_duration_ms = args.time * MILLI_SEC

    if args.qps:
        qps = float(args.qps)
        settings.server_target_qps = qps
        settings.offline_expected_qps = qps

    if count_override:
        settings.min_query_count = count
        settings.max_query_count = count

    if args.samples_per_query:
        settings.multi_stream_samples_per_query = args.samples_per_query
    if args.max_latency:
        settings.server_target_latency_ns = int(args.max_latency * NANO_SEC)
        settings.multi_stream_expected_latency_ns = int(args.max_latency * NANO_SEC)
    
    
    performance_sample_count = (
        args.performance_sample_count
        if args.performance_sample_count
        else min(count, 1)
    )
    sut = lg.ConstructSUT(runner.issue_queries, runner.flush_queries)
    qsl = lg.ConstructQSL(
        count, performance_sample_count, ds.load_query_samples, ds.unload_query_samples
    )
    

    log.info("starting {}".format(scenario))
    result_dict = {"scenario": str(scenario)}
    
    lg.StartTestWithLogSettings(sut, qsl, settings, log_settings, audit_config)
    
    print("start test finish")
    if args.accuracy:
        post_proc.finalize(result_dict, ds, output_dir=args.output)
        final_results["accuracy_results"] = result_dict
        post_proc.save_images(saved_images_ids, ds)

    print("before runner finish")
    runner.finish()
    print("after runner finish")
    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)

    #
    # write final results
    #
    if args.output:
        with open("results.json", "w") as f:
            json.dump(final_results, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
