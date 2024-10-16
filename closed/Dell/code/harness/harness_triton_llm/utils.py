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
from collections import namedtuple
from typing import Union, Optional
import numpy as np
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import json

LlmConfig = namedtuple('LlmConfig', 'model_name, min_output_len, max_output_len, beam_width, end_token_id, pad_token_id, streaming')


def construct_triton_input_data(name: str, shape: Union[list, tuple, None] = None,
                                dtype: str = "INT32", tensor: Optional[np.array] = None, protocol: str = "http") -> Union[httpclient.InferInput, grpcclient.InferInput]:
    if shape is None:
        shape = tensor.shape

    assert protocol in ["http", "grpc"]
    if protocol == "http":
        infer_input = httpclient.InferInput(name=name, shape=list(shape), datatype=dtype)
        if tensor is not None:
            infer_input.set_data_from_numpy(tensor)
    else:
        infer_input = grpcclient.InferInput(name=name, shape=list(shape), datatype=dtype)
        if tensor is not None:
            infer_input.set_data_from_numpy(tensor)

    return infer_input


def get_gptj_triton_inputs(tensor_input_len: np.array, tensor_input_ids: np.array,
                           llm_config: LlmConfig, protocol: str = "http"):
    """
    For a single sample, accepts (len, ids) and returns list of triton input objects
    """

    assert tensor_input_len.shape == (1, 1), "Triton only supports BS==1 for now (TRTLLM-298)"
    assert tensor_input_ids.shape[0] == 1, "Triton only supports BS==1 for now (TRTLLM-298)"

    ip_len = tensor_input_len[0, 0]
    tensor_input_ids = tensor_input_ids[:, :ip_len].reshape(1, -1)
    tensor_output_len = np.array([llm_config.max_output_len], dtype="int32").reshape([1, 1])
    tensor_beam_width = np.array([llm_config.beam_width], dtype="int32").reshape([1, 1])
    tensor_streaming = np.array([llm_config.streaming], dtype="bool").reshape([1, 1])
    tensor_end_id = np.array([llm_config.end_token_id], dtype="int32").reshape([1, 1])
    tensor_pad_id = np.array([llm_config.pad_token_id], dtype="int32").reshape([1, 1])

    input_ids = construct_triton_input_data(name="input_ids", tensor=tensor_input_ids, protocol=protocol)
    input_lengths = construct_triton_input_data(name="input_lengths", tensor=tensor_input_len, protocol=protocol)
    request_output_len = construct_triton_input_data(name="request_output_len", tensor=tensor_output_len, protocol=protocol)
    beam_width = construct_triton_input_data(name="beam_width", tensor=tensor_beam_width, protocol=protocol)
    end_id = construct_triton_input_data(name="end_id", tensor=tensor_end_id, protocol=protocol)
    pad_id = construct_triton_input_data(name="pad_id", tensor=tensor_pad_id, protocol=protocol)
    streaming = construct_triton_input_data(name="streaming", tensor=tensor_streaming, dtype="BOOL", protocol=protocol)

    return [input_ids, input_lengths, request_output_len, beam_width, end_id, pad_id, streaming]


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
