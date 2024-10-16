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
from importlib import import_module
from os import PathLike
from pathlib import Path
from typing import Optional, Dict

import tensorrt as trt
import onnx
import re

from nvmitten.constants import Precision
from nvmitten.nvidia.builder import (TRTBuilder,
                                     CalibratableTensorRTEngine,
                                     MLPerfInferenceEngine,
                                     ONNXNetwork,
                                     LegacyBuilder)
from nvmitten.pipeline import Operation
from nvmitten.utils import dict_get, logging

from code.common.fields import Fields
from code.common.mitten_compat import ArgDiscarder
from code.common.systems.system_list import DETECTED_SYSTEM, SystemClassifications


Subnetwork = ONNXNetwork.Subnetwork
RN50Calibrator = import_module("code.resnet50.tensorrt.calibrator").RN50Calibrator
RN50GraphSurgeon = import_module("code.resnet50.tensorrt.rn50_graphsurgeon").RN50GraphSurgeon
ResNet50Component = import_module("code.resnet50.tensorrt.constants").ResNet50Component

resnet50_tactic_dict = {
    "res3a_branch2a \+ res3a_branch2a_relu.*": "0x266a23a6dae5e9dd",
    "res3a_branch1.*": "0xdf9672025c2e4e0b",
    "res3a_branch2b \+ res3a_branch2b_relu.*": "0x11e97a6f7b62ebde",
    "res3b_branch2a \+ res3b_branch2a_relu.*": "0x7b6ce4f355d80159",
    "res3b_branch2b \+ res3b_branch2b_relu.*": "0x11e97a6f7b62ebde",
    "res3c_branch2a \+ res3c_branch2a_relu.*": "0x7b6ce4f355d80159",
    "res3c_branch2b \+ res3c_branch2b_relu.*": "0x11e97a6f7b62ebde",
    "res3d_branch2a \+ res3d_branch2a_relu.*": "0x7b6ce4f355d80159",
    "res3d_branch2b \+ res3d_branch2b_relu.*": "0x11e97a6f7b62ebde",
    "res4a_branch2a \+ res4a_branch2a_relu.*": "0x743ccad8b4fb4cdc",
    "res4a_branch1.*": "0x42ae598d672de6e2",
    "res4a_branch2b \+ res4a_branch2b_relu.*": "0xdaed974d12d937ad",
    "res4b_branch2a \+ res4b_branch2a_relu.*": "0xd470dbb02604d85e",
    "res4b_branch2b \+ res4b_branch2b_relu.*": "0x5d63ba6b5cfeb06f",
    "res4c_branch2a \+ res4c_branch2a_relu.*": "0xd470dbb02604d85e",
    "res4c_branch2b \+ res4c_branch2b_relu.*": "0x5d63ba6b5cfeb06f",
    "res4d_branch2a \+ res4d_branch2a_relu.*": "0xd470dbb02604d85e",
    "res4d_branch2b \+ res4d_branch2b_relu.*": "0x5d63ba6b5cfeb06f",
    "res4e_branch2a \+ res4e_branch2a_relu.*": "0xd470dbb02604d85e",
    "res4e_branch2b \+ res4e_branch2b_relu.*": "0x5d63ba6b5cfeb06f",
    "res4f_branch2a \+ res4f_branch2a_relu.*": "0xd470dbb02604d85e",
    "res4f_branch2b \+ res4f_branch2b_relu.*": "0x5d63ba6b5cfeb06f",
    "res5a_branch1.*": "0xdaed974d12d937ad",
    "res5a_branch2a \+ res5a_branch2a_relu.*": "0x8d507197d88f65f7",
    "res5a_branch2b \+ res5a_branch2b_relu.*": "0xdaed974d12d937ad",
    "res5b_branch2a \+ res5b_branch2a_relu.*": "0xdaed974d12d937ad",
    "res5b_branch2b \+ res5b_branch2b_relu.*": "0xdaed974d12d937ad",
    "res5c_branch2a \+ res5c_branch2a_relu.*": "0xdaed974d12d937ad",
    "res5c_branch2b \+ res5c_branch2b_relu.*": "0xdaed974d12d937ad",
    "fc_replaced.*": "0x4ce968916c7f46ae"
}


class HopperTacticSelector(trt.IAlgorithmSelector):
    def select_algorithms(self, ctx, choices):
        print("\nselect algorithms: " + ctx.name)
        resnet50_layer_pattern = re.compile('|'.join(resnet50_tactic_dict))
        if re.search(resnet50_layer_pattern, ctx.name):
            print("Matched")
            for layer_regex, tactic_id in resnet50_tactic_dict.items():
                if re.match(layer_regex, ctx.name):
                    filtered_idxs = [idx for idx, choice in enumerate(choices) if choice.algorithm_variant.tactic == int(tactic_id, 16)]
                    to_ret = filtered_idxs
                    print("Filtered id: ", tactic_id)
        else:
            to_ret = [idx for idx, _ in enumerate(choices)]
        return to_ret

    def report_algorithms(self, ctx, choices):
        pass

# MLPINF-2560: resnet50 builder is already broken on Orin in v4.1 prior to the GBS change. Fix in Thor submission


class ResNet50EngineBuilder(CalibratableTensorRTEngine,
                            TRTBuilder,
                            MLPerfInferenceEngine,
                            ArgDiscarder):
    """ResNet50 end to end base builder class.
    """

    def __init__(self,
                 # TODO: Legacy value - Remove after refactor is done.
                 config_ver: str = "default",
                 # TODO: This should be a relative path within the ScratchSpace.
                 model_path: str = "build/models/ResNet50/resnet50_v1.onnx",
                 # Override the default values
                 calib_data_map: PathLike = Path("data_maps/imagenet/cal_map.txt"),
                 cache_file: PathLike = Path("code/resnet50/tensorrt/calibrator.cache"),
                 # Benchmark specific values
                 component: str = None,
                 batch_size: int = 1,
                 disable_beta1_smallk: bool = False,
                 energy_aware_kernels: bool = False,
                 **kwargs):
        super().__init__(calib_data_map=calib_data_map,
                         cache_file=cache_file,
                         **kwargs)
        self.config_ver = config_ver
        self.model_path = model_path
        self.component = component
        self.batch_size = batch_size
        self.disable_beta1_smallk = disable_beta1_smallk
        self.energy_aware_kernels = energy_aware_kernels

        self.device_type = "dla" if self.dla_enabled else "gpu"
        self.int8_dyn_range = (-128, 127)

    def set_calibrator(self, image_dir):
        if self.precision != Precision.INT8:
            return

        self.calibrator = RN50Calibrator(calib_batch_size=self.calib_batch_size,
                                         calib_max_batches=self.calib_max_batches,
                                         force_calibration=self.force_calibration,
                                         cache_file=self.cache_file,
                                         calib_data_map=self.calib_data_map,
                                         image_dir=image_dir)

    def get_subnetwork(self):
        # base end to end resnet builder does not have subnetwork
        if self.__class__ is ResNet50EngineBuilder:
            subnet = None
        else:
            subnet = RN50GraphSurgeon.subnetwork_map.get(self.component)
            if not subnet:
                raise ValueError(f"subnetwork_name: {self.component} not supported by rn50_graphsurgeon")
        return subnet

    def create_network(self, builder: trt.Builder):
        network = super().create_network(builder)
        compute_sm = DETECTED_SYSTEM.get_compute_sm()

        # Parse from ONNX file
        parser = trt.OnnxParser(network, self.logger)
        rn50_gs = RN50GraphSurgeon(self.model_path,
                                   self.precision,
                                   self.cache_file,
                                   compute_sm,
                                   self.device_type,
                                   self.need_calibration,
                                   disable_beta1_smallk=self.disable_beta1_smallk)
        subnet = self.get_subnetwork()

        model = rn50_gs.create_onnx_model(subnetwork=subnet)
        success = parser.parse(model.SerializeToString())
        if not success:
            err_desc = parser.get_error(0).desc()
            raise RuntimeError(f"ResNet50 onnx model processing failed! Error: {err_desc}")

        # Check input and output tensor names
        if subnet:
            if subnet.inputs:
                actual = {network.get_input(i).name for i in range(network.num_inputs)}
                expected = {tens.name for tens in subnet.inputs}
                assert actual == expected, f"Subnetwork input mismatch: Got: {actual}, Expected: {expected}"
            if subnet.outputs:
                actual = {network.get_output(i).name for i in range(network.num_outputs)}
                expected = {tens.name for tens in subnet.outputs}
                assert actual == expected, f"Subnetwork output mismatch: Got: {actual}, Expected: {expected}"

        self.apply_subnetwork_io_types(network)
        return network

    def apply_subnetwork_io_types(self, network: trt.INetworkDefinition):
        """Applies I/O dtype and formats for subnetwork inputs and outputs to the tensorrt.INetworkDefinition.

        Note: Currently in Mitten, ONNXNetwork.Subnetwork does not support tensor data format or dynamic range
        specification, as well as expected devices. Hence, we take the subnetwork name instead of the subnetwork
        description itself as an argument.

        Args:
            network (tensorrt.INetworkDefinition): The network generated from the builder.
            subnetwork_name (str): The name or ID of the subnetwork
        """
        tensor_in = network.get_input(0)

        self._discard_topk_output_value(network)
        self._set_tensor_format(tensor_in, use_dla=self.dla_enabled)

    def _discard_topk_output_value(self, network: trt.INetworkDefinition):
        """Unmark topk_layer_output_value, just leaving topk_layer_output_index.
        """
        assert network.num_outputs == 2, "Two outputs expected"
        assert network.get_output(0).name == "topk_layer_output_value",\
            f"unexpected tensor: {network.get_output(0).name}"
        assert network.get_output(1).name == "topk_layer_output_index",\
            f"unexpected tensor: {network.get_output(1).name}"
        logging.info(f"Unmarking output: {network.get_output(0).name}")
        network.unmark_output(network.get_output(0))

    def _set_tensor_format(self,
                           tensor: trt.ITensor,
                           use_dla: bool = False,
                           tensor_format: Optional[trt.TensorFormat] = None,
                           dynamic_range: Optional[Tuple[int, int]] = None):
        """Set input tensor dtype and format.

        Args:
            input_tensor_name (str): The tensor to modify
            use_dla (bool): If True, uses DLA input formats if applicable. (Default: False)
            tensor_format (trt.TensorFormat): Overrides the tensor format to set the input to. If not set, uses
                                              self.input_format. (Default: None)
            dynamic_range (Tuple[int, int]): A tuple of length 2 in the format [min_value (inclusive), max_value
                                             (inclusive)]. This argument is ignored if the input tensor is not in INT8
                                             precision. (Default: None)
        """
        # Apply dynamic ranges for INT8 inputs
        assert (len(self.input_dtype) == 1)
        if self.input_dtype[0] == "int8":
            tensor.dtype = trt.int8
            if dynamic_range is not None:
                tensor.dynamic_range = dynamic_range

        if not tensor_format:
            assert len(self.input_format) == 1
            # Set the same format as the input data if not specified
            if self.input_format[0] == "linear":
                tensor_format = trt.TensorFormat.LINEAR
            elif self.input_format[0] == "chw4":
                # WAR for DLA reformat bug in https://nvbugs/3713387:
                # For resnet50, inputs dims are [3, 224, 224].
                # For those particular dims, CHW4 == DLA_HWC4, so can use same CHW4 data for both GPU and DLA engine.
                # By lying to TRT and saying input is DLA_HWC4, we elide the pre-DLA reformat layer.
                if use_dla:
                    tensor_format = trt.TensorFormat.DLA_HWC4
                else:
                    tensor_format = trt.TensorFormat.CHW4
        tensor.allowed_formats = 1 << int(tensor_format)

    def create_builder_config(self, *args, **kwargs):
        builder_config = super().create_builder_config(*args, **kwargs)
        builder_config.int8_calibrator = self.calibrator
        if self.energy_aware_kernels:
            builder_config.algorithm_selector = HopperTacticSelector()
        return builder_config


# MLPINF-2560: resnet50 builder is already broken on Orin in v4.1 prior to the GBS change. Fix in Thor submission
class ResNet50BackboneEngineBuilder(ResNet50EngineBuilder):
    """ResNet50 backbone builder class.
    """

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args,
                         **kwargs)

    def apply_subnetwork_io_types(self, network: trt.INetworkDefinition):
        """Applies I/O dtype and formats for class specific components' inputs and outputs to the tensorrt.INetworkDefinition.

        Args:
            network (tensorrt.INetworkDefinition): The network generated from the builder.
        """
        tensor_in = network.get_input(0)
        tensor_out = network.get_output(0)

        self._set_tensor_format(tensor_in, use_dla=self.dla_enabled, dynamic_range=self.int8_dyn_range)
        self._set_tensor_format(tensor_out, dynamic_range=self.int8_dyn_range)


# MLPINF-2560: resnet50 builder is already broken on Orin in v4.1 prior to the GBS change. Fix in Thor submission
class ResNet50TopKEngineBuilder(ResNet50EngineBuilder):
    """ResNet50 topK builder class.
    """

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args,
                         **kwargs)

    def apply_subnetwork_io_types(self, network: trt.INetworkDefinition):
        """Applies I/O dtype and formats for class specific components' inputs and outputs to the tensorrt.INetworkDefinition.

        Args:
            network (tensorrt.INetworkDefinition): The network generated from the builder.
        """
        tensor_in = network.get_input(0)

        self._discard_topk_output_value(network)
        self._set_tensor_format(tensor_in, dynamic_range=self.int8_dyn_range)


# MLPINF-2560: resnet50 builder is already broken on Orin in v4.1 prior to the GBS change. Fix in Thor submission
class ResNet50PreRes2EngineBuilder(ResNet50EngineBuilder):
    """ResNet50 preres2 builder class.
    """

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args,
                         **kwargs)

    def apply_subnetwork_io_types(self, network: trt.INetworkDefinition):
        """Applies I/O dtype and formats for class specific components' inputs and outputs to the tensorrt.INetworkDefinition.

        Args:
            network (tensorrt.INetworkDefinition): The network generated from the builder.
        """
        tensor_in = network.get_input(0)
        tensor_out = network.get_output(0)

        self._set_tensor_format(tensor_in, dynamic_range=self.int8_dyn_range)
        self._set_tensor_format(tensor_out, tensor_format=trt.TensorFormat.CHW32)


# MLPINF-2560: resnet50 builder is already broken on Orin in v4.1 prior to the GBS change. Fix in Thor submission
class ResNet50PreRes3EngineBuilder(ResNet50EngineBuilder):
    """ResNet50 preres3 builder class.
    """

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args,
                         **kwargs)

    def apply_subnetwork_io_types(self, network: trt.INetworkDefinition):
        """Applies I/O dtype and formats for class specific components' inputs and outputs to the tensorrt.INetworkDefinition.

        Args:
            network (tensorrt.INetworkDefinition): The network generated from the builder.
        """
        tensor_in = network.get_input(0)
        tensor_out = network.get_output(0)

        self._set_tensor_format(tensor_in, dynamic_range=self.int8_dyn_range)
        self._set_tensor_format(tensor_out, tensor_format=trt.TensorFormat.CHW32)


# MLPINF-2560: resnet50 builder is already broken on Orin in v4.1 prior to the GBS change. Fix in Thor submission
class ResNet50Res2_3EngineBuilder(ResNet50EngineBuilder):
    """ResNet50 res2res3 builder class.
    """

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args,
                         **kwargs)

    def apply_subnetwork_io_types(self, network: trt.INetworkDefinition):
        """Applies I/O dtype and formats for class specific components' inputs and outputs to the tensorrt.INetworkDefinition.

        Args:
            network (tensorrt.INetworkDefinition): The network generated from the builder.
        """
        tensor_in = network.get_input(0)
        tensor_out = network.get_output(0)

        self._set_tensor_format(tensor_in, tensor_format=trt.TensorFormat.CHW32)
        self._set_tensor_format(tensor_out, tensor_format=trt.TensorFormat.CHW32)


# MLPINF-2560: resnet50 builder is already broken on Orin in v4.1 prior to the GBS change. Fix in Thor submission
class ResNet50Res3EngineBuilder(ResNet50EngineBuilder):
    """ResNet50 res3 builder class.
    """

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args,
                         **kwargs)

    def apply_subnetwork_io_types(self, network: trt.INetworkDefinition):
        """Applies I/O dtype and formats for class specific components' inputs and outputs to the tensorrt.INetworkDefinition.

        Args:
            network (tensorrt.INetworkDefinition): The network generated from the builder.
        """
        tensor_in = network.get_input(0)
        tensor_out = network.get_output(0)

        self._set_tensor_format(tensor_in, tensor_format=trt.TensorFormat.CHW32)
        self._set_tensor_format(tensor_out, tensor_format=trt.TensorFormat.CHW32)


# MLPINF-2560: resnet50 builder is already broken on Orin in v4.1 prior to the GBS change. Fix in Thor submission
class ResNet50PostRes3EngineBuilder(ResNet50EngineBuilder):
    """ResNet50 postres3 builder class.
    """

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args,
                         **kwargs)

    def apply_subnetwork_io_types(self, network: trt.INetworkDefinition):
        """Applies I/O dtype and formats for class specific components' inputs and outputs to the tensorrt.INetworkDefinition.

        Args:
            network (tensorrt.INetworkDefinition): The network generated from the builder.
        """
        tensor_in = network.get_input(0)

        self._discard_topk_output_value(network)
        self._set_tensor_format(tensor_in, tensor_format=trt.TensorFormat.CHW32)


class ResNet50EngineBuilderOp(Operation,
                              ArgDiscarder):
    COMPONENT_BUILDER_MAP = {
        ResNet50Component.ResNet50: ResNet50EngineBuilder,
        ResNet50Component.Backbone: ResNet50BackboneEngineBuilder,
        ResNet50Component.TopK: ResNet50TopKEngineBuilder,
        ResNet50Component.PreRes2: ResNet50PreRes2EngineBuilder,
        ResNet50Component.PreRes3: ResNet50PreRes3EngineBuilder,
        ResNet50Component.Res2Res3: ResNet50Res2_3EngineBuilder,
        ResNet50Component.Res3: ResNet50Res3EngineBuilder,
        ResNet50Component.PostRes3: ResNet50PostRes3EngineBuilder,
    }

    @classmethod
    def immediate_dependencies(cls):
        # TODO: Integrate dataset scripts as deps
        return None

    def __init__(self,
                 *args,
                 # Benchmark specific values
                 batch_size: Dict[ResNet50Component, int] = None,
                 **kwargs):
        """Creates a ResNet50EngineBuilderOp.

        Args:
            batch_size (Dict[str, int]): Component and its batch size to build the engine for)
        """
        super().__init__(*args, **kwargs)
        if not batch_size:
            logging.warning(f"No batch_size dict provided for ResNet50EngineBuilderOp. Setting to default value {ResNet50Component.ResNet50 : 1}")
            batch_size = {ResNet50Component.ResNet50: 1}
        self.builders = []
        for component, component_batch_size in batch_size.items():
            builder = ResNet50EngineBuilderOp.COMPONENT_BUILDER_MAP[component](*args, component=component.valstr(), batch_size=component_batch_size, **kwargs)
            self.builders.append(builder)

    def run(self, scratch_space, dependency_outputs):
        for builder in self.builders:
            # Set up INT8 calibration
            builder.set_calibrator(scratch_space.path / "preprocessed_data" / "imagenet" / "ResNet50" / "fp32")

            builder_config = builder.create_builder_config()
            network = builder.create_network(builder.builder)
            engine_dir = builder.engine_dir(scratch_space)
            engine_name = builder.engine_name(builder.device_type,
                                              builder.batch_size,
                                              builder.precision,
                                              builder.component,
                                              tag=builder.config_ver)
            engine_fpath = engine_dir / engine_name

            builder(builder.batch_size, engine_fpath, network)


class ResNet50(LegacyBuilder):
    """Temporary spoofing class to wrap around Mitten to adhere to the old API.
    """

    def __init__(self, args):
        super().__init__(ResNet50EngineBuilderOp(**args))
