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

import os
from abc import ABC, abstractmethod
import tensorrt as trt
from code.common import logging, dict_get
from code.common.constants import TRT_LOGGER, Scenario
from code.common.fields import Fields
from code.common.constants import Benchmark


class AbstractBuilder(ABC):
    """Interface base class for calibrating and building engines."""

    @abstractmethod
    def build_engines(self):
        """
        Builds the engine using assigned member variables as parameters.
        """
        pass

    @abstractmethod
    def calibrate(self):
        """
        Performs INT8 calibration using variables as parameters. If INT8 calibration is not supported for the Builder,
        then this method should print a message saying so and return immediately.
        """
        pass


class MultiBuilder(AbstractBuilder):
    """
    MultiBuilder allows for building multiple engines sequentially. As an example, RNN-T has multiple components, each of
    which have separate engines, which we would like to abstract away.
    """

    def __init__(self, builders, args):
        """
        MultiBuilder takes in a list of Builder classes and args to be passed to these Builders.
        """
        self.builders = list(builders)
        self.args = args

    def build_engines(self):
        for b in self.builders:
            b(self.args).build_engines()

    def calibrate(self):
        for b in self.builders:
            b(self.args).calibrate()
