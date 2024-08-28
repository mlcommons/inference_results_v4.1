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


import functools
import os
import sys
import venv

from typing import Dict, Union, List
from pathlib import Path
from importlib import import_module

from code.common import run_command

from .constants import DLRMv2_BUILD_VENV_PATH


class DLRMv2Injector:
    """
    provides dependency injection from dlrm build-venv via annotations.
    does so by overriding globals with injected modules at invocation.

    Implementation
    --------------
    1. keeps track of injections to make with `create_injection`
    2. overrides function.__globals__ at invocation


    Usage
    -----
    create injector. can have multiple instances, eg: one for each file needing it.
    >>> injector = DLRMv2Injector(venv_path=PATH_TO_DLRM_VENV)

    create injections which will be fulfilled at invocation
    >>> DEFAULT_CAT_NAMES, INT_FEATURE_COUNT = injector.create_injection('torch.datasets.criteo', ['DEFAULT_CAT_NAMES', 'INT_FEATURE_COUNT'])

    utilize injections by wrapping scope with provided annotation.
    injections will not work if annotation is not used for wrapping function scope.
    >>> @injector.inject_dlrm_dependencies()
    ... def consume_injections(): print(DEFAULT_CAT_NAMES, INT_FEATURE_COUNT)

    can now use annotated function as normal. all injections will be made at this invocation.
    >>> consume_injections()

    Notes
    -----
    injections do not propagate down the callstack. all functions which use injections need to be wrapped with annotation individually.
    """

    class Injection:
        def __init__(self, module: str, name: str):
            self.module = module
            self.name = name

        def __str__(self) -> str:
            return f'<DLRMv2Injector.Injection name: {self.name}, module: {self.module}, obj: [loaded at runtime]>'

        @property
        def is_flat(self) -> bool:
            return self.module == self.name

    def __init__(self, venv_path: os.PathLike = DLRMv2_BUILD_VENV_PATH):
        self.venv_path = Path(venv_path)
        self.injections: Dict[str, DLRMv2Injector.Injection] = {}

    def create_injection(self, module: str, name: Union[str | List[str]] = None) -> Union[Injection | List[Injection]]:
        """
        create injections to be fulfilled later.
        can be used as underlying object itself, but value is only held at runtime.

        Usage
        -----
        >>> from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, INT_FEATURE_COUNT

        becomes:

        >>> DEFAULT_CAT_NAMES, INT_FEATURE_COUNT = injector.create_injection('torch.datasets.criteo', ['DEFAULT_CAT_NAMES', 'INT_FEATURE_COUNT'])

        Parameters
        ----------
        module: import path for injection
        name: name(s) to import from module
        """

        if (name is None) or isinstance(name, str):
            name = name or module  # import module itself
            self.injections[name] = DLRMv2Injector.Injection(module, name)
            return self.injections[name]

        else:
            for name_ in name:
                self.injections[name_] = DLRMv2Injector.Injection(module, name_)
            return [self.injections[name_] for name_ in name]

    def inject_dlrm_dependencies(self):
        """
        inject dlrm dependencies at runtime.
        this will override globals for the wrapped function call with injections.
        """

        # setup build-virtualenv if needed
        # this venv contains the additional/alternative requirements to what is already installed in container
        if not self.venv_path.exists():
            build_context = os.getenv('BUILD_CONTEXT')
            venv.create(self.venv_path, with_pip=True)
            run_command(f"{self.venv_path}/bin/pip install -q -r /work/code/dlrm-v2/tensorrt/build_requirements.{build_context}.txt")

            if build_context == 'x86_64':
                run_command(f"{self.venv_path}/bin/pip uninstall -q -y torch numpy")

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                original_globals = dict(func.__globals__)
                original_path = sys.path.copy()

                # add venv path primary
                path = str(self.venv_path / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/')
                sys.path.insert(0, str(path))

                # inject dependencies as globals into wrapped code
                for name, injection in self.injections.items():
                    module = import_module(injection.module)
                    func.__globals__[name] = module if injection.is_flat else module.__dict__[name]

                try:
                    return func(*args, **kwargs)

                finally:
                    sys.path = original_path
                    func.__globals__.clear()
                    func.__globals__.update(original_globals)

            return wrapper
        return decorator
