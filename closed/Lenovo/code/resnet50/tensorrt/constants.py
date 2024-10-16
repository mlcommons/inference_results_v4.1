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
from code.common.constants import AliasedName, AliasedNameEnum


class ResNet50Component(AliasedNameEnum):
    """Names of supported Benchmarks for resnet50."""

    ResNet50: AliasedName = AliasedName("resnet50", ("resnet",))
    Backbone: AliasedName = AliasedName("backbone", ("resnet50-backbone", "resnet-backbone",))
    TopK: AliasedName = AliasedName("topk", ("resnet50-topk", "resnet-topk",))
    PreRes2: AliasedName = AliasedName("preres2", ("resnet50-preres2", "resnet-preres2",))
    PreRes3: AliasedName = AliasedName("preres3", ("resnet50-preres3", "resnet-preres3",))
    Res2Res3: AliasedName = AliasedName("res2_3", ("resnet50-res2_3", "resnet-res2_3",))
    Res3: AliasedName = AliasedName("res3", ("resnet50-res3", "resnet-res3",))
    PostRes3: AliasedName = AliasedName("postres3", ("resnet50-postres3", "resnet-postres3",))
