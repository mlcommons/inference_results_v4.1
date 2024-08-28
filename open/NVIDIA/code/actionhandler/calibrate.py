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

from typing import Any, Dict, List

from code import get_benchmark
from code.actionhandler.base import ActionHandler
from code.common import logging, dict_eq
from code.common.constants import *

from nvmitten.nvidia.builder import CalibratableTensorRTEngine


class CalibrateHandler(ActionHandler):
    """Handles the Calibrate action."""

    def __init__(self, benchmark_conf):
        """Creates a new ActionHandler for Calibrate

        Args:
            benchmark_conf (Dict[str, Any]): The benchmark configuration in dictionary form (Legacy behavior)
        """
        super().__init__(Action.Calibrate)

        self.benchmark_conf = benchmark_conf
        self.system = benchmark_conf["system"]
        self.benchmark = benchmark_conf["benchmark"]
        self.scenario = benchmark_conf["scenario"]
        self.workload_setting = benchmark_conf["workload_setting"]

    def setup(self):
        """Called once before handle().
        """
        logging.info(f"Generating calibration cache for Benchmark \"{self.benchmark.valstr()}\"")
        self.benchmark_conf["dla_core"] = None  # Cannot calibrate on DLA
        self.benchmark_conf["force_calibration"] = True
        self.benchmark_conf["batch_size"] = self.benchmark_conf["gpu_batch_size"]

    def cleanup(self, success: bool):
        """Called after handle(), regardless if it errors.
        """
        return

    def handle(self):
        """Run the action.
        """
        if self.benchmark == Benchmark.ResNet50:
            logging.info("Calibration for RN50 with TRT >= 8.5 may result in a bad calibration cache.")
            logging.info("Please Use the v3.0 calibration cache instead:")
            logging.info("https://raw.githubusercontent.com/mlcommons/inference_results_v3.0/main/closed/NVIDIA/code/resnet50/tensorrt/calibrator.cache")
            return True
        b = get_benchmark(self.benchmark_conf)

        # use default builder to calibrate whole network
        mitten_bulder_op = b.mitten_builder
        if isinstance(mitten_bulder_op, CalibratableTensorRTEngine):  # non GSB style, will be removed in stage 2
            mitten_builder = b.mitten_builder
        else:  # GBS style
            mitten_builder = b.mitten_builder.builders[0]

        if not isinstance(mitten_builder, CalibratableTensorRTEngine):
            logging.info(f"{mitten_builder} is not instance of CalibratableTensorRTEngine. Skipping calibrate.")

        else:
            old_fields = dict()

            def _cache_and_set(attr, val):
                old_fields[attr] = getattr(mitten_builder, attr)
                setattr(mitten_builder, attr, val)

            # Unlike the old legacy API, we don't need to call .clear_cache(), since the calibrator is created in
            # TRTBuilder.run() instead of __init__.
            # Note that after calibration, TensorRT will still build the engine. In this case, we set the batch size to 1 to
            # make it go faster, but I'm not sure how to skip it.
            _cache_and_set("force_calibration", True)
            _cache_and_set("batch_size", 1)
            _cache_and_set("create_profiles", mitten_builder.calibration_profiles)

            # Do not run the full Mitten pipeline yet. Invoke run manually.
            mitten_bulder_op.run(b.legacy_scratch, None)

            # Restore old values
            for attr, val in old_fields.items():
                setattr(mitten_builder, attr, val)

        return True

    def handle_failure(self):
        """Called after handle() if it errors.
        """
        raise RuntimeError("Calibration failed!")
