#!/usr/bin/env python3
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
from os import PathLike
from pathlib import Path
from typing import Optional, Dict
from importlib import import_module

import importlib.util
import time
import os
import subprocess
import sys

from nvmitten.constants import Precision
from nvmitten.nvidia.builder import (TRTBuilder,
                                     MLPerfInferenceEngine,
                                     LegacyBuilder)
from nvmitten.pipeline import Operation
from nvmitten.utils import logging

from code.common.mitten_compat import ArgDiscarder
from code.common.systems.system_list import SystemClassifications
GPTJComponent = import_module("code.gptj.tensorrt.constants").GPTJComponent


class GPTJ6BEngineBuilder(TRTBuilder,
                          MLPerfInferenceEngine,
                          ArgDiscarder):
    """GPT-J offloads the engine building to the implementation in the TRT-LLM examples. This class still is a sublass
    of nvmitten.nvidia.builder.TRTBuilder solely for consistency and to use the same class initializer.
    """

    def __init__(self,
                 # TODO: Legacy value - Remove after refactor is done.
                 config_ver: str = "default",
                 # TODO: This should be a relative path within the ScratchSpace.
                 model_path: str = "build/models/GPTJ-6B/checkpoint-final",
                 fp8_quant_model_path: str = "build/models/GPTJ-6B/fp8-quantized-ammo/GPTJ-FP8-quantized",
                 trt_llm_path: str = "build/TRTLLM",
                 # Override the normal default values
                 workspace_size: int = 60 << 30,
                 # Benchmark specific values
                 batch_size: int = 16,
                 use_fp8: bool = False,
                 use_inflight_batching: bool = False,
                 max_num_tokens: int = 16384,
                 int4_quant_model_path: str = "build/models/GPTJ-6B/orin-w4a16-awq",
                 **kwargs):
        super().__init__(workspace_size=workspace_size,
                         **kwargs)

        self.config_ver = config_ver
        self.batch_size = batch_size
        self.model_path = model_path
        self.use_fp8 = use_fp8
        self.fp8_quant_model_path = Path(fp8_quant_model_path)
        self.use_inflight_batching = use_inflight_batching
        self.dtype = "fp8" if self.use_fp8 else "fp16"
        self.need_quantization = False
        # required for generating engine
        self.max_num_tokens = max_num_tokens
        self.int4_quant_model_path = Path(int4_quant_model_path)

        # https://gitlab-master.nvidia.com/ftp/tekit/-/tree/main/examples/gptj#fp8-post-training-quantization
        if not SystemClassifications.is_orin():
            if self.use_fp8:
                if not self.fp8_quant_model_path.exists():
                    raise FileNotFoundError(f"Could not locate GPT-J fp8 quantized checkpoint model path: ({self.fp8_quant_model_path}). Please check README.md")
                    # TODO If the default model does not exist, build a quantized model locally
                    # self.need_quantization = True
            else:
                raise NotImplementedError(f"Only fp8 supported as of now. Set precision to FP16 and use_fp8 to True. If you need to perform quantization, please refer to README.md")
        else:
            if not self.int4_quant_model_path.exists():
                raise FileNotFoundError(f"Could not locate GPT-J fp16 checkpoint model path: ({self.int4_quant_model_path}). Please check README.md")

        self.trt_llm_path = Path(trt_llm_path)

        if self.precision == "fp8":
            raise NotImplementedError(f"To enable FP8 precision, set precision to FP16 and use_fp8 to True.")
        elif self.precision != Precision.FP16:
            raise NotImplementedError(f"Precision {self.precision} is not supported yet.")

    def build_quantized_model(self):
        """
        Use AMMO to build the quantized FP8 model for TRTLLM usage
        """
        quantize_script_path = self.trt_llm_path / "examples/gptj/quantize.py"
        if not quantize_script_path.exists():
            raise FileNotFoundError(f"Could not locate GPT-J quantize script ({quantize_script_path}), please run `make clone_trt_llm`")

        quantize_dir = self.fp8_quant_model_path.parent
        quantize_dir.mkdir(parents=True, exist_ok=True)
        flags = [
            f"--model_dir={self.model_path}",
            "--dtype=float16",
            "--qformat=fp8",
            f"--export_path={quantize_dir}",
            "--calib_size=512"
        ]
        quantize_cmd = [sys.executable, str(quantize_script_path.absolute())] + flags
        logging.info(f"Building GPTJ FP8 quantization model in {quantize_dir}.")
        logging.info(f"Command: {' '.join(quantize_cmd)}")

        stdout_log = self.fp8_quant_model_path.with_suffix(".stdout")
        stderr_log = self.fp8_quant_model_path.with_suffix(".stderr")

        tik = time.time()
        ret = subprocess.run(quantize_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        tok = time.time()

        # Save stdout and stderr logs
        with stdout_log.open(mode='w') as f:
            f.write(ret.stdout)
        with stderr_log.open(mode='w') as f:
            f.write(ret.stderr)

        if ret.returncode != 0:
            logging.error(ret.stderr)
            raise RuntimeError(f"Quantization recipe failed. Logs dumped to {stderr_log}.")

        logging.info(f"Quantized model completes in {tok-tik}s. Saved to {self.fp8_quant_model_path}")

    def create_network(self):
        logging.info(f"GPTJ network is created implicitly in TRTLLM")

    def build_engine(self,
                     network: trt.INetworkDefinition,
                     builder_config: trt.IBuilderConfig,
                     batch_size: int,
                     engine_fpath: PathLike):
        """Builds the engine via the GPT-J builder provided in the TRT-LLM examples.

        Args:
            network: Unused.
            builder_config: Unused.
            batch_size (int): Batch size to build the engine for.
            engine_fpath (PathLike): Location to save the engine file(s) to.
        """
        if not importlib.util.find_spec("tensorrt_llm"):
            raise ModuleNotFoundError("Cannot import tensorrt_llm module. Please run `make build_trt_llm`.")

        if self.need_quantization:
            self.build_quantized_model()

        build_script = Path("tensorrt_llm/commands/build.py")
        build_script_path = self.trt_llm_path / build_script
        if not build_script_path.exists():
            raise FileNotFoundError(f"Could not locate TRTLLM build script ({build_script_path}), please run `make clone_trt_llm`")

        engine_fpath = Path(engine_fpath)
        if engine_fpath.is_file():
            logging.warning(f"{engine_fpath} already exists. This file will be overwritten")
        engine_dir = engine_fpath.parent
        engine_dir.mkdir(parents=True, exist_ok=True)

        flags = ["--gpt_attention_plugin=float16",
                 f"--max_batch_size={self.batch_size}",
                 "--max_input_len=1919",
                 "--max_seq_len=2047",
                 "--max_beam_width=4",
                 f"--max_num_tokens={self.max_num_tokens}",
                 f"--output_dir={str(engine_dir.absolute())}",
                 "--context_fmha=enable",
                 "--remove_input_padding=enable",
                 ]
        if SystemClassifications.is_orin():
            flags += ["--paged_kv_cache=enable",
                      "--gemm_plugin=float16",
                      f"--checkpoint_dir={str(self.int4_quant_model_path.absolute())}", ]
        else:
            flags += ["--paged_kv_cache=enable",
                      f"--checkpoint_dir={str(self.fp8_quant_model_path.absolute())}", ]

        build_cmd = [sys.executable, "-m", '.'.join(build_script.with_suffix('').parts)] + flags

        custom_env = os.environ.copy()
        custom_env["TRTLLM_DISABLE_XQA_JIT"] = "1"

        logging.info(f"Building GPTJ engine in {engine_dir}.")
        logging.info(f"Command executing in build/TRTLLM dir: {' '.join(build_cmd)}")

        stdout_log = engine_fpath.with_suffix(".stdout")
        stderr_log = engine_fpath.with_suffix(".stderr")

        tik = time.time()
        ret = subprocess.run(build_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd="build/TRTLLM", env=custom_env)
        tok = time.time()

        # Save stdout and stderr logs
        with stdout_log.open(mode='w') as f:
            f.write(ret.stdout)
        with stderr_log.open(mode='w') as f:
            f.write(ret.stderr)

        if ret.returncode != 0:
            logging.error(ret.stderr)
            raise RuntimeError(f"Engine build failed. Logs dumped to {engine_dir}.")

        logging.info(f"Engine build complete in {tok-tik}s. Saved to {engine_fpath}")


class GPTJ6BEngineBuilderOp(Operation,
                            ArgDiscarder):
    COMPONENT_BUILDER_MAP = {
        GPTJComponent.GPTJ: GPTJ6BEngineBuilder,
    }

    @classmethod
    def immediate_dependencies(cls):
        # TODO: Integrate dataset scripts as deps
        return None

    def __init__(self,
                 *args,
                 # Benchmark specific values
                 batch_size: Dict[GPTJComponent, int] = None,
                 **kwargs):
        """Creates a GPTJ6BEngineBuilderOp.

        Args:
            batch_size (Dict[str, int]): Component and its batch size to build the engine for)
        """
        super().__init__(*args, **kwargs)
        if not batch_size:
            logging.warning("No batch_size dict provided for GPTJ6BEngineBuilderOp. Setting to default value {GPTJComponent.GPTJ : 1}")
            batch_size = {GPTJComponent.GPTJ: 1}
        self.builders = []
        for component, component_batch_size in batch_size.items():
            builder = GPTJ6BEngineBuilderOp.COMPONENT_BUILDER_MAP[component](*args, batch_size=component_batch_size, **kwargs)
            self.builders.append(builder)

    def run(self, scratch_space, dependency_outputs):
        for builder in self.builders:
            engine_dir = builder.engine_dir(scratch_space)
            # We distinguish engines by the dir name for TRTLLM
            engine_dir = engine_dir / f"bs{builder.batch_size}-{builder.config_ver}"
            # For TRT-LLM, the engine name is fixed.
            engine_name = "rank0.engine"
            engine_fpath = engine_dir / engine_name

            builder.build_engine(None, None, builder.batch_size, engine_fpath)


class GPTJ6B(LegacyBuilder):
    """Temporary spoofing class to wrap around Mitten to adhere to the old API.
    """

    def __init__(self, args):
        super().__init__(GPTJ6BEngineBuilderOp(**args))
