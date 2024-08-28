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
from typing import Union
import numpy as np
from cuda import cuda, cudart


def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t")

    if len(cuda_ret) > 1:
        return cuda_ret[1]

    return None


class CUDAPythonContext:
    '''global cuda-python context'''
    default_device: cuda.CUdevice = None
    context: cuda.CUcontext = None

    @staticmethod
    def initialize(device_id: int = 0):
        '''init and create context if needed'''
        if CUDAPythonContext.context is None:
            CUASSERT(cuda.cuInit(0))
            CUDAPythonContext.default_device = CUASSERT(cuda.cuDeviceGet(device_id))
            CUDAPythonContext.context = CUASSERT(cuda.cuCtxCreate(0, CUDAPythonContext.default_device))

    @staticmethod
    def reset():
        '''cleans up context if needed'''
        CUASSERT(cuda.cuCtxDestroy(CUDAPythonContext.context))
        CUDAPythonContext.default_device = None
        CUDAPythonContext.context = None


class CUDAPythonScopedContext:
    '''scoped cuda-python context'''

    def __init__(self, device_id: int = 0):
        CUASSERT(cuda.cuInit(0))
        self.device = CUASSERT(cuda.cuDeviceGet(device_id))

    def __enter__(self):
        self.context = CUASSERT(cuda.cuCtxCreate(0, self.device))
        return self

    def __exit__(self, *_):
        CUASSERT(cuda.cuCtxDestroy(self.context))


def np_from_pointer(pointer: int, nptype: type, shape: Union[tuple, list], copy: bool = False, read_only_flag: bool = False):
    """
    NOTE(vir): from https://stackoverflow.com/a/56755422

    Generates numpy array from memory address
    https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.interface.html

    Parameters
    ----------
    pointer : int
        Memory address

    nptype : type
        Numpy datatype for tensor elements

    shape : Union[tuple|list]
        Shape of array.

    copy : bool
        Copy array.  Default False

    read_only_flag : bool
        Read only array.  Default False.
    """

    np_to_typestr = {
        # NOTE(vir): see https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.interface.html#__array_interface__
        # Example: little-endian FP16 = '<f2'
        np.float32: "<f4",
        np.float16: "<f2",
        np.int32: "<i4",
    }

    assert nptype in np_to_typestr, "Format not yet supported!"
    buff = {
        'data': (pointer, read_only_flag),
        'typestr': np_to_typestr[nptype],
        'shape': shape
    }

    class numpy_holder():
        pass

    holder = numpy_holder()
    holder.__array_interface__ = buff
    return np.array(holder, copy=copy)
