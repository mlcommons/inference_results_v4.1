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


import os
from collections import defaultdict
from importlib import import_module
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import distributed as torch_distrib

import tensorrt as trt

from nvmitten.pipeline import Operation
from nvmitten.nvidia.builder import CalibratableTensorRTEngine, LegacyBuilder, MLPerfInferenceEngine, TRTBuilder

from polygraphy import func
from polygraphy.logger import G_LOGGER
from polygraphy.backend.trt import Calibrator, CreateConfig, CreateNetwork, EngineFromNetwork, Profile, TrtRunner

from code.common import logging
from code.common.constants import TRT_LOGGER
from code.common.mitten_compat import ArgDiscarder
from code.plugin import load_trt_plugin_by_network
from code.common.systems.system_list import SystemClassifications
if SystemClassifications.is_soc():
    raise NotImplementedError("SOC is not supported for dlrmv2")

from .calibrator import DLRMv2Calibrator
from .scripts.gen_frequency_data import gen_frequency_data
from .utils import DLRMv2Injector
from .criteo import CRITEO_SYNTH_MULTIHOT_N_EMBED_PER_FEATURE, CRITEO_SYNTH_MULTIHOT_SIZES, CriteoDay23Dataset

DLRMv2Component = import_module("code.dlrm-v2.tensorrt.constants").DLRMv2Component
load_trt_plugin_by_network("dlrmv2")

# NOTE(vir): use dependency injection to satisfy torchrec deps
injector = DLRMv2Injector()
Snapshot = injector.create_injection("torchsnapshot", "Snapshot")
EmbeddingBagCollection = injector.create_injection("torchrec", "EmbeddingBagCollection")
get_local_size = injector.create_injection("torchrec.distributed.comm", "get_local_size")
EmbeddingBagConfig = injector.create_injection("torchrec.modules.embedding_configs", "EmbeddingBagConfig")
HeuristicalStorageReservation = injector.create_injection("torchrec.distributed.planner.storage_reservations", "HeuristicalStorageReservation")
DistributedModelParallel, get_default_sharders = injector.create_injection("torchrec.distributed.model_parallel", ["DistributedModelParallel", "get_default_sharders"])
EmbeddingShardingPlanner, Topology = injector.create_injection("torchrec.distributed.planner", ["EmbeddingShardingPlanner", "Topology"])
DLRM_DCN, DLRMTrain = injector.create_injection("torchrec.models.dlrm", ["DLRM_DCN", "DLRMTrain"])
DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES, INT_FEATURE_COUNT, CAT_FEATURE_COUNT = injector.create_injection(
    "torchrec.datasets.criteo",
    ["DEFAULT_CAT_NAMES", "DEFAULT_INT_NAMES", "INT_FEATURE_COUNT", "CAT_FEATURE_COUNT"])


class DLRMv2_Model:
    def __init__(self,
                 model_path: os.PathLike = "/home/mlperf_inf_dlrmv2/model/model_weights",
                 num_embeddings_per_feature: int = CRITEO_SYNTH_MULTIHOT_N_EMBED_PER_FEATURE,
                 embedding_dim: int = 128,
                 dcn_num_layers: int = 3,
                 dcn_low_rank_dim: int = 512,
                 dense_arch_layer_sizes: List[int] = (512, 256, 128),
                 over_arch_layer_sizes: List[int] = (1024, 1024, 512, 256, 1),
                 load_ckpt_on_gpu: bool = False):
        self.model_path = Path(model_path)
        self.state_dict_path = self.model_path.parent / 'mini_state_dict.pt'

        self.num_embeddings_per_feature = list(num_embeddings_per_feature)
        self.embedding_dim = embedding_dim
        self.dcn_num_layers = dcn_num_layers
        self.dcn_low_rank_dim = dcn_low_rank_dim
        self.dense_arch_layer_sizes = list(dense_arch_layer_sizes)
        self.over_arch_layer_sizes = list(over_arch_layer_sizes)

        if load_ckpt_on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.distrib_backend = "nccl"

        else:
            self.device = torch.device("cpu")
            self.distrib_backend = "gloo"

        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        torch_distrib.init_process_group(backend=self.distrib_backend, rank=0, world_size=1)

        # cache model to avoid re-loading
        self.model = None

    def load_state_dict(self):
        if self.state_dict_path.exists():
            # if possible dont load full pytorch model, only state dict. this is faster
            logging.info(f'Loading State Dict from: {self.state_dict_path}')
            return torch.load(str(self.state_dict_path))

        else:
            # load model from sharded files using pytorch & cache state dict for subsequent runs
            self.model = self.load_model()
            return self.model.state_dict()

    @injector.inject_dlrm_dependencies()
    def load_model(self, return_snapshot: bool = False):
        logging.info('Loading Model...')
        self.embedding_bag_configs = [
            EmbeddingBagConfig(name=f"t_{feature_name}",
                               embedding_dim=self.embedding_dim,
                               num_embeddings=self.num_embeddings_per_feature[feature_idx],
                               feature_names=[feature_name])
            for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
        ]

        # create model
        self.embedding_bag_collection = EmbeddingBagCollection(tables=self.embedding_bag_configs, device=torch.device("meta"))
        torchrec_dlrm_config = DLRM_DCN(embedding_bag_collection=self.embedding_bag_collection, dense_in_features=len(DEFAULT_INT_NAMES),
                                        dense_arch_layer_sizes=self.dense_arch_layer_sizes,
                                        over_arch_layer_sizes=self.over_arch_layer_sizes,
                                        dcn_num_layers=self.dcn_num_layers,
                                        dcn_low_rank_dim=self.dcn_low_rank_dim,
                                        dense_device=self.device)
        torchrec_dlrm_model = DLRMTrain(torchrec_dlrm_config)

        # distribute the model
        planner = EmbeddingShardingPlanner(
            topology=Topology(local_world_size=get_local_size(), world_size=torch_distrib.get_world_size(), compute_device=self.device.type),
            storage_reservation=HeuristicalStorageReservation(percentage=0.05)
        )
        plan = planner.collective_plan(torchrec_dlrm_model, get_default_sharders(), torch_distrib.GroupMember.WORLD)
        model = DistributedModelParallel(module=torchrec_dlrm_model,
                                         device=self.device,
                                         plan=plan)

        # load weights
        snapshot = Snapshot(path=str(self.model_path))
        snapshot.restore(app_state={"model": model})
        model.eval()

        # remove embeddings from state dict
        minified_sd = model.state_dict().copy()
        for key in [key for key in minified_sd.keys() if 'embedding_bags' in key]:
            del minified_sd[key]

        # save a stripped state dict for easier loading
        torch.save(minified_sd, str(self.state_dict_path))

        if return_snapshot:
            return model, snapshot

        else:
            return model

    @injector.inject_dlrm_dependencies()
    def get_embedding_weight(self, cat_feature_idx: int):
        assert cat_feature_idx < len(DEFAULT_CAT_NAMES)

        # load model if not already loaded
        if not self.model:
            self.model = self.load_model()

        embedding_bag_state = self.model.module.model.sparse_arch.embedding_bag_collection.state_dict()
        key = f"embedding_bags.t_cat_{cat_feature_idx}.weight"
        out = torch.zeros(embedding_bag_state[key].metadata().size, device=self.device)
        embedding_bag_state[key].gather(0, out=out)
        return out

    @injector.inject_dlrm_dependencies()
    def dump_embedding_weights(self, save_dir: os.PathLike):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        def int8_quantize(mega_table):
            # compute scales
            mults = np.ndarray(shape=(CAT_FEATURE_COUNT))
            scales = np.ndarray(shape=(CAT_FEATURE_COUNT))
            for id, table in enumerate(mega_table):
                maxAbsVal = abs(max(table.max(), table.min(), key=abs))
                scales[id] = maxAbsVal / 127.0
                mults[id] = 1.0 / scales[id]

            # multiply scales, symmetric quantization
            mega_table_int8 = []
            for id, table in enumerate(mega_table):
                mega_table_int8.append(np.minimum(np.maximum(np.rint(table * mults[id]), -127), 127).astype(np.int8))

            return (np.vstack(mega_table_int8).reshape(-1).astype(np.int8), scales.astype(np.float32))

        # collect mega table
        mega_table = []
        for i in range(len(DEFAULT_CAT_NAMES)):
            weight = self.get_embedding_weight(i).cpu()
            mega_table.append(weight.numpy())

        # compute mega_table and scales for all support precisions
        precision_to_tensor = {
            'fp32': (np.vstack(mega_table).reshape(-1).astype(np.float32), None),
            'fp16': (np.vstack(mega_table).reshape(-1).astype(np.float16), None),
            'int8': int8_quantize(mega_table)
        }

        # save all mega_tables and scales
        for precision, (table, scales) in precision_to_tensor.items():
            table_path = save_dir / f"mega_table_{precision}.npy"
            logging.info(f'Saving mega_table [{precision}]: {table_path}')

            with open(table_path, 'wb') as table_file:
                np.save(table_file, table)

            if scales is not None:
                scales_path = save_dir / f'mega_table_scales.npy'
                logging.info(f'Saving mega_table_scales [{precision}]: {scales_path}')

                with open(scales_path, 'wb') as scales_file:
                    np.save(scales_file, scales)

    @injector.inject_dlrm_dependencies()
    def load_embeddings(self, from_dir: os.PathLike):
        embedding_bag_configs = [
            EmbeddingBagConfig(name=f"t_{feature_name}",
                               embedding_dim=self.embedding_dim,
                               num_embeddings=self.num_embeddings_per_feature[feature_idx],
                               feature_names=[feature_name])
            for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
        ]

        embedding_bag_collection = EmbeddingBagCollection(tables=embedding_bag_configs,
                                                          device=self.device)

        # torchrec 0.3.2 does not support init_fn as a EmbeddingBagConfig parameter.
        # Manually implement it here.
        for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES):
            with open(Path(from_dir) / f"embed_feature_{feature_idx}.weight.pt", 'rb') as f:
                dat = torch.load(f, map_location=self.device)
            with torch.no_grad():
                embedding_bag_collection.embedding_bags[f"t_{feature_name}"].weight.copy_(dat)
        return embedding_bag_collection


class DLRMv2Arch:
    """Loose representation of the DLRMv2 architecture based on TorchRec source:
    https://github.com/pytorch/torchrec/blob/main/torchrec/models/dlrm.py

    The components of the model are as follows:

    1. SparseArch (Embedding table, isolated into DLRMv2_Model)
    2. DenseArch (Bottom MLP)
    3. InteractionDCNArch (DCNv2, or sometimes referred to as interactions network / layer)
    4. OverArch (Top MLP + final linear layer)
    """

    def __init__(self,
                 state_dict,
                 bot_mlp_depth: int = 3,
                 crossnet_depth: int = 3,
                 top_mlp_depth: int = 4):
        self.bot_mlp_depth = bot_mlp_depth
        self.bottom_mlp = self.create_bot_mlp(state_dict)

        self.crossnet_depth = crossnet_depth
        self.crossnet = self.create_crossnet(state_dict)

        self.top_mlp_depth = top_mlp_depth
        self.top_mlp = self.create_top_mlp(state_dict)

        self.final_linear = self.create_final_linear(state_dict)

    def create_bot_mlp(self, state_dict):
        """ Bottom MLP keys
        model.dense_arch.model._mlp.0._linear.bias
        model.dense_arch.model._mlp.0._linear.weight
        model.dense_arch.model._mlp.1._linear.bias
        model.dense_arch.model._mlp.1._linear.weight
        model.dense_arch.model._mlp.2._linear.bias
        model.dense_arch.model._mlp.2._linear.weight
        """
        conf = defaultdict(dict)
        for i in range(self.bot_mlp_depth):
            key_prefix = f"model.dense_arch.model._mlp.{i}._linear."
            conf[i]["weight"] = state_dict[key_prefix + "weight"]
            conf[i]["bias"] = state_dict[key_prefix + "bias"]
        return conf

    def create_crossnet(self, state_dict):
        """ DCNv2 crossnet is based on torchrec.modules.crossnet.LowRankCrossNet:
            - https://pytorch.org/torchrec/torchrec.modules.html#torchrec.modules.crossnet.LowRankCrossNet
            - https://github.com/pytorch/torchrec/blob/42c55844d29343c644521e810597fd67017eac8f/torchrec/modules/crossnet.py#L90

        Keys:
        model.inter_arch.crossnet.V_kernels.0
        model.inter_arch.crossnet.V_kernels.1
        model.inter_arch.crossnet.V_kernels.2
        model.inter_arch.crossnet.W_kernels.0
        model.inter_arch.crossnet.W_kernels.1
        model.inter_arch.crossnet.W_kernels.2
        model.inter_arch.crossnet.bias.0
        model.inter_arch.crossnet.bias.1
        model.inter_arch.crossnet.bias.2
        """
        conf = defaultdict(dict)
        for i in range(self.crossnet_depth):
            V = f"model.inter_arch.crossnet.V_kernels.{i}"
            W = f"model.inter_arch.crossnet.W_kernels.{i}"
            bias = f"model.inter_arch.crossnet.bias.{i}"
            conf[i]['V'] = state_dict[V]
            conf[i]['W'] = state_dict[W]
            conf[i]["bias"] = state_dict[bias]
        return conf

    def create_top_mlp(self, state_dict):
        """ Top MLP keys
        model.over_arch.model.0._mlp.0._linear.bias
        model.over_arch.model.0._mlp.0._linear.weight
        model.over_arch.model.0._mlp.1._linear.bias
        model.over_arch.model.0._mlp.1._linear.weight
        model.over_arch.model.0._mlp.2._linear.bias
        model.over_arch.model.0._mlp.2._linear.weight
        model.over_arch.model.0._mlp.3._linear.bias
        model.over_arch.model.0._mlp.3._linear.weight
        """
        conf = defaultdict(dict)
        for i in range(self.top_mlp_depth):
            key_prefix = f"model.over_arch.model.0._mlp.{i}._linear."
            conf[i]["weight"] = state_dict[key_prefix + "weight"]
            conf[i]["bias"] = state_dict[key_prefix + "bias"]
        return conf

    def create_final_linear(self, state_dict):
        """ Probability reduction linear layer keys
        model.over_arch.model.1.bias
        model.over_arch.model.1.weight
        """
        conf = {
            "weight": state_dict["model.over_arch.model.1.weight"],
            "bias": state_dict["model.over_arch.model.1.bias"],
        }
        return conf


class DLRMv2TRTNetwork:
    def __init__(self,
                 network: trt.INetworkDefinition,
                 batch_size: int,
                 verbose: bool,
                 model_path: os.PathLike,
                 embeddings_path: os.PathLike,
                 dense_dtype: str,
                 sparse_dtype: str,
                 bot_mlp_precision: str,
                 embeddings_precision: str,
                 interaction_op_precision: str,
                 top_mlp_precision: str,
                 final_linear_precision: str,
                 embedding_weights_on_gpu_part: float,
                 use_row_frequencies_opt: bool,
                 row_frequencies_npy_filepath: os.PathLike):
        self.network = network
        self.batch_size = batch_size
        self.verbose = verbose
        self.logger = TRT_LOGGER
        self.logger.min_severity = trt.Logger.VERBOSE if self.verbose else trt.Logger.INFO

        self.model_path = Path(model_path)
        self.embeddings_path = Path(embeddings_path)

        # embeddings_path is created if needed in dump_embedding_weights
        assert model_path.exists()

        precision_str_to_type = {'fp32': trt.float32, 'fp16': trt.float16, 'int8': trt.int8, 'int32': trt.int32}
        self.dense_dtype = precision_str_to_type[dense_dtype]
        self.sparse_dtype = precision_str_to_type[sparse_dtype]
        self.bot_mlp_precision = precision_str_to_type[bot_mlp_precision]
        self.embeddings_precision = precision_str_to_type[embeddings_precision]
        self.interaction_op_precision = precision_str_to_type[interaction_op_precision]
        self.top_mlp_precision = precision_str_to_type[top_mlp_precision]
        self.final_linear_precision = precision_str_to_type[final_linear_precision]

        self.embedding_weights_on_gpu_part = embedding_weights_on_gpu_part
        self.use_row_frequencies_opt = use_row_frequencies_opt
        self.row_frequencies_npy_filepath = Path(row_frequencies_npy_filepath)

        self.mega_table_npy_file = self.embeddings_path / f'mega_table_{embeddings_precision}.npy'
        self.mega_table_scales_npy_file = self.embeddings_path / 'mega_table_scales.npy'

        self.pprint_network_precision_configs()

        self.initialize_arch()
        self.initialize_network()

    def pprint_network_precision_configs(self):
        precision_type_to_str = {
            trt.float32: 'fp32',
            trt.float16: 'fp16',
            trt.int8: 'int8',
            trt.int32: 'int32'
        }

        logging.info(f'Network Config:'
                     f'\n\tdense_dtype: {precision_type_to_str[self.dense_dtype]}'
                     f'\n\tsparse_dtype: {precision_type_to_str[self.sparse_dtype]}'
                     f'\n\tbot_mlp_precision: {precision_type_to_str[self.bot_mlp_precision]}'
                     f'\n\tembeddings_precision: {precision_type_to_str[self.embeddings_precision]}'
                     f'\n\tinteraction_op_precision: {precision_type_to_str[self.interaction_op_precision]}'
                     f'\n\ttop_mlp_precision: {precision_type_to_str[self.top_mlp_precision]}'
                     f'\n\tfinal_linear_precision: {precision_type_to_str[self.final_linear_precision]}')

    def parse_calibration(self, cache_path: os.PathLike = 'code/dlrm-v2/tensorrt/calibrator.cache'):
        cache_path = Path(cache_path)
        assert cache_path.exists(), "calibration cache missing, int8 engine cannot be built"

        with open(cache_path, 'rb') as f:
            lines = f.read().decode('ascii').splitlines()

        calibration_dict = {}
        for line in lines:
            split = line.split(':')
            if len(split) != 2:
                continue

            tensor = split[0]
            drange = np.uint32(int(split[1], 16)).view(np.float32).item() * np.float32(127.0)
            calibration_dict[tensor] = drange

        return calibration_dict

    def initialize_arch(self):
        # load state dict and arch
        dlrm_model = DLRMv2_Model(model_path=self.model_path)
        state_dict = dlrm_model.load_state_dict()

        # load embeddings mega table and scales, create if needed
        if not self.mega_table_npy_file.exists() or not self.mega_table_scales_npy_file.exists():
            logging.info("Generating missing embedding files...")
            dlrm_model.dump_embedding_weights(self.embeddings_path)
        else:
            logging.info("Found embedding mega_table and scales file.")

        self.arch = DLRMv2Arch(state_dict)
        self.embedding_size = dlrm_model.embedding_dim

    @injector.inject_dlrm_dependencies()
    def initialize_network(self):
        # create inputs
        # numerical input from harness:             [-1, 13, 1, 1]
        # sparse input from harness:                [-1, total_hotness]
        numerical_input = self.network.add_input('numerical_input', self.dense_dtype, (-1, INT_FEATURE_COUNT, 1, 1,))
        sparse_input = self.network.add_input('sparse_input', self.sparse_dtype, (-1, sum(CRITEO_SYNTH_MULTIHOT_SIZES)))

        # create bottom_mlp                         [-1, 13, 1, 1] -> [-1, 128, 1, 1]
        bot_mlp = self._build_mlp(self.arch.bottom_mlp, numerical_input, INT_FEATURE_COUNT, 'bot_mlp', precision=self.bot_mlp_precision)

        # create dense_input                        [-1, 128, 1, 1] -> [-1, 128]
        squeeze_dense = self.network.add_shuffle(bot_mlp.get_output(0))
        squeeze_dense.reshape_dims = (-1, self.embedding_size)
        squeeze_dense.precision = self.bot_mlp_precision
        squeeze_dense.name = 'bot_mlp.squeeze'
        squeeze_dense.get_output(0).name = 'bot_mlp.squeeze.output'
        dense_input = squeeze_dense.get_output(0)

        # NOTE(vir): cant set op.precision as sparse input is always int32
        # create embedding lookup plugin            [-1, 128], [-1, table_hotness] -> [-1, 3456]
        dlrm_embedding_lookup_plugin = self.get_dlrmv2_embedding_lookup_plugin(self.bot_mlp_precision, self.sparse_dtype, self.embeddings_precision)
        embedding_op = self.network.add_plugin_v2([dense_input, sparse_input], dlrm_embedding_lookup_plugin)
        embedding_op.name = "dlrmv2_embedding_lookup"
        embedding_op.get_output(0).name = "dlrmv2_embedding_lookup.output"

        # create interaction op                     [-1, 3456] -> [-1, 3456]
        interaction_input = embedding_op.get_output(0)
        interaction_op = self._build_interaction_op(self.arch.crossnet, interaction_input, precision=self.interaction_op_precision)

        # create top mlp                            [-1, 3456] -> [-1, 256]
        top_mlp_input = interaction_op.get_output(0)
        top_mlp = self._build_mlp(self.arch.top_mlp, top_mlp_input, top_mlp_input.shape[1], 'top_mlp', precision=self.top_mlp_precision)

        # create final linear layer                 [-1, 256] -> [-1, 1]
        final_linear_input = top_mlp.get_output(0)
        final_linear = self._build_linear(self.arch.final_linear, final_linear_input, final_linear_input.shape[1], 'final_linear', add_relu=False, precision=self.final_linear_precision)

        # NOTE(vir): sigmoid needs to be in fp32 precision
        # create sigmoid output layer
        # input from final linear:                  [-1, 1] -> [-1, 1]
        sigmoid_input = final_linear.get_output(0)
        sigmoid_layer = self.network.add_activation(sigmoid_input, trt.ActivationType.SIGMOID)
        sigmoid_layer.name = "sigmoid"
        sigmoid_layer.get_output(0).name = "sigmoid.output"

        # mark fp32 output
        sigmoid_output = sigmoid_layer.get_output(0)
        sigmoid_output.dtype = trt.float32
        self.network.mark_output(sigmoid_output)

    def _build_mlp(self,
                   config,
                   in_tensor,
                   in_channels,
                   name_prefix,
                   use_conv_for_fc=False,
                   precision=trt.float32):

        for index, state in config.items():
            layer = self._build_linear(state,
                                       in_tensor,
                                       in_channels,
                                       f'{name_prefix}_{index}',
                                       use_conv_for_fc=use_conv_for_fc,
                                       precision=precision)

            in_channels = state['weight'].shape[::-1][-1]
            in_tensor = layer.get_output(0)

        return layer

    def _build_linear(self,
                      state,
                      in_tensor,
                      in_channels,
                      name,
                      add_relu=True,
                      use_conv_for_fc=False,
                      precision=trt.float32):

        weights = state['weight'].numpy()
        bias = state['bias'].numpy().reshape(1, -1)

        shape = weights.shape[::-1]
        out_channels = shape[-1]

        if use_conv_for_fc:
            layer = self.network.add_convolution_nd(in_tensor, out_channels, (1, 1), weights, bias)
            layer.precision = precision
            layer.name = name + '.conv'
            layer.get_output(0).name = name + '.conv.output'

        else:
            if len(in_tensor.shape) == 4:
                squeeze_dense = self.network.add_shuffle(in_tensor)
                squeeze_dense.reshape_dims = (-1, in_tensor.shape[1])
                squeeze_dense.precision = precision
                squeeze_dense.name = name + '.squeeze'
                squeeze_dense.get_output(0).name = name + '.squeeze.output'
                in_tensor = squeeze_dense.get_output(0)

            w_tens = self.network.add_constant(weights.shape, weights)
            w_tens.precision = precision
            w_tens.name = name + '.mm.w'
            w_tens.get_output(0).name = name + '.mm.w.output'

            b_tens = self.network.add_constant(bias.shape, bias)
            b_tens.precision = precision
            b_tens.name = name + '.b.w'
            b_tens.get_output(0).name = name + '.b.w.output'

            wx = self.network.add_matrix_multiply(in_tensor, trt.MatrixOperation.NONE, w_tens.get_output(0), trt.MatrixOperation.TRANSPOSE)
            wx.precision = precision
            wx.name = name + '.mm'
            wx.get_output(0).name = name + '.mm.output'

            bwx = layer = self.network.add_elementwise(wx.get_output(0), b_tens.get_output(0), trt.ElementWiseOperation.SUM)
            bwx.precision = precision
            bwx.name = name + '.b'
            bwx.get_output(0).name = name + '.b.output'

        if add_relu:
            layer = self.network.add_activation(layer.get_output(0), trt.ActivationType.RELU)
            layer.precision = precision
            layer.name = name + ".relu"
            layer.get_output(0).name = name + ".relu.output"

        return layer

    @injector.inject_dlrm_dependencies()
    def _build_interaction_op(self, config, x, precision=trt.float32, use_conv=True, use_explicit_qdq=False):
        # From LowRankCrossNet docs:
        # https://pytorch.org/torchrec/torchrec.modules.html#torchrec.modules.crossnet.LowRankCrossNet
        # x_next = x_0 * (matmul(W_curr, matmul(V_curr, x_curr)) + bias_curr) + x_curr

        # enable explicit qdq on tensor
        # TODO(vir): read scale from cache based on node name
        def insert_qdq(input, scale=1.0):
            if not use_explicit_qdq:
                return input

            scale = self.network.add_constant([1], np.array([scale], np.float32)).get_output(0)
            quant = self.network.add_quantize(input, scale).get_output(0)
            dequant = self.network.add_dequantize(quant, scale).get_output(0)
            return dequant

        if use_conv:
            # unsqueeze input [-1, 3456] -> [-1, 3456, 1, 1]
            unsqueeze = self.network.add_shuffle(x)
            unsqueeze.reshape_dims = (-1, (CAT_FEATURE_COUNT * self.embedding_size) + self.embedding_size, 1, 1)
            unsqueeze.precision = unsqueeze.precision if use_explicit_qdq else precision
            unsqueeze.name = 'interaction.unsqueeze'
            unsqueeze.get_output(0).name = 'interaction.unsqueeze.output'
            x = unsqueeze.get_output(0)

        x = insert_qdq(x)
        x0 = x

        for index, state in config.items():
            V = state['V'].numpy()  # 512 x 3456
            W = state['W'].numpy()  # 3456 x 512
            b = state['bias'].numpy()  # 3456,

            # set weights
            if use_conv:
                V = V.reshape(*V.shape, 1, 1)
                W = W.reshape(*W.shape, 1, 1)
                b = b.reshape(1, b.shape[0], 1, 1)

            else:
                V_tens = self.network.add_constant(V.shape, V)
                V_tens.precision = V_tens.precision if use_explicit_qdq else precision
                V_tens.name = f'interaction.V_{index}'
                V_tens.get_output(0).name = f'interaction.V_{index}.output'
                V_tens = V_tens.get_output(0)
                V_tens = insert_qdq(V_tens)

                W_tens = self.network.add_constant(W.shape, W)
                W_tens.precision = W_tens.precision if use_explicit_qdq else precision
                W_tens.name = f'interaction.W_{index}'
                W_tens.get_output(0).name = f'interaction.W_{index}.output'
                W_tens = W_tens.get_output(0)
                W_tens = insert_qdq(W_tens)

                b_tens = self.network.add_constant([1, b.shape[0]], b)
                b_tens.precision = b_tens.precision if use_explicit_qdq else precision
                b_tens.name = f'interaction.b_{index}'
                b_tens.get_output(0).name = f'interaction.b_{index}.output'
                b_tens = b_tens.get_output(0)
                b_tens = insert_qdq(b_tens)

            # set operations
            vx = self.network.add_convolution_nd(x, V.shape[0], (1, 1), V)                                           \
                if use_conv else                                                                                     \
                self.network.add_matrix_multiply(x, trt.MatrixOperation.NONE, V_tens, trt.MatrixOperation.TRANSPOSE)
            vx.precision = vx.precision if use_explicit_qdq else precision
            vx.name = f'interaction.vx_{index}'
            vx.get_output(0).name = f'interaction.vx_{index}.output'
            vx = vx.get_output(0)
            vx = insert_qdq(vx)

            wvx = self.network.add_convolution_nd(vx, W.shape[0], (1, 1), W, b)                                       \
                if use_conv else                                                                                      \
                self.network.add_matrix_multiply(vx, trt.MatrixOperation.NONE, W_tens, trt.MatrixOperation.TRANSPOSE)
            wvx.precision = wvx.precision if use_explicit_qdq else precision
            wvx.name = f'interaction.wvx_{index}'
            wvx.get_output(0).name = f'interaction.wvx_{index}.output'
            wvx = wvx.get_output(0)
            wvx = insert_qdq(wvx)

            if use_conv:
                inner = wvx  # bias integrated in conv layer

            else:
                inner = self.network.add_elementwise(wvx, b_tens, trt.ElementWiseOperation.SUM)
                inner.precision = inner.precision if use_explicit_qdq else precision
                inner.name = f'interaction.inner_{index}'
                inner.get_output(0).name = f'interaction.inner_{index}.output'
                inner = inner.get_output(0)
                inner = insert_qdq(inner)

            left_term = self.network.add_elementwise(inner, x0, trt.ElementWiseOperation.PROD)
            left_term.precision = left_term.precision if use_explicit_qdq else precision
            left_term.name = f'interaction.left_term_{index}'
            left_term.get_output(0).name = f'interaction.left_term_{index}.output'
            left_term = left_term.get_output(0)
            left_term = insert_qdq(left_term)

            x_ = self.network.add_elementwise(left_term, x, trt.ElementWiseOperation.SUM)
            x_.precision = x.precision if use_explicit_qdq else precision
            x_.name = f'interaction.out_{index}'
            x_.get_output(0).name = f'interaction.out_{index}.output'

            # port for next layer
            x = x_.get_output(0)
            x = insert_qdq(x)

        if use_conv:
            # squeeze output [-1, 3456, 1, 1] -> [-1, 3456]
            squeeze = self.network.add_shuffle(x)
            squeeze.reshape_dims = (-1, (CAT_FEATURE_COUNT * self.embedding_size) + self.embedding_size)
            squeeze.precision = squeeze.precision if use_explicit_qdq else precision
            squeeze.name = 'interaction.squeeze'
            squeeze.get_output(0).name = 'interaction.squeeze.output'
            output = squeeze

        else:
            output = x_

        return output

    @injector.inject_dlrm_dependencies()
    def get_dlrmv2_embedding_lookup_plugin(self,
                                           dense_input_type: trt.DataType,
                                           sparse_input_type: trt.DataType,
                                           output_type: trt.DataType):
        """Create a plugin layer for the DLRMv2 Embedding Lookup plugin and return it. """

        pluginName = "DLRMv2_EMBEDDING_LOOKUP_TRT"
        embeddingRows = sum(CRITEO_SYNTH_MULTIHOT_N_EMBED_PER_FEATURE)
        tableOffsets = np.concatenate(([0], np.cumsum(CRITEO_SYNTH_MULTIHOT_N_EMBED_PER_FEATURE).astype(np.int32)[:-1])).astype(np.int32)
        tableHotness = np.array(CRITEO_SYNTH_MULTIHOT_SIZES).astype(np.int32)
        totalHotness = sum(CRITEO_SYNTH_MULTIHOT_SIZES)

        # NOTE(vir): WAR: int8 dense input type for plugin does not pass accuracy
        # trivial perf advantage of using (+ fixing) int8 dense input type
        if dense_input_type == trt.int8:
            logging.warning('Plugin with dense input type int8 does not pass accuracy. Using fp16 instead.')
            dense_input_type = trt.float16

        # NOTE(vir): mirror of setup in dlrmv2EmbeddingLookupPlugin.cpp
        supported_format_combinations = {
            (trt.float32, trt.int32, trt.float32): 0,
            (trt.float16, trt.int32, trt.float16): 1,
            (trt.float16, trt.int32, trt.int8): 2,
        }

        format_combination = (dense_input_type, sparse_input_type, output_type)
        reducedPrecisionIO = supported_format_combinations.get(format_combination, None)
        assert reducedPrecisionIO is not None, f"plugin does not support format combination: {format_combination}"

        # NOTE(vir): generate row frequencies table if needed, read in plugin init
        if not self.row_frequencies_npy_filepath.exists():
            logging.info("Row frequencies table does not exist. Generating...")
            gen_frequency_data(out_file=self.row_frequencies_npy_filepath)
            logging.info("Generated row frequencies table.")
        else:
            logging.info("Found row frequencies table.")

        if self.use_row_frequencies_opt:
            assert output_type != trt.int8, "Check usage, int8 embedding mega_table fits completely on most NVIDIA mlperf-inference skus"
            logging.warning("Enabled use_row_frequencies_opt. Make sure embedding rows on host > 0.")

        plugin = None
        for plugin_creator in trt.get_plugin_registry().plugin_creator_list:
            if plugin_creator.name == pluginName:
                embeddingSize_field = trt.PluginField("embeddingSize", np.array([self.embedding_size], dtype=np.int32), trt.PluginFieldType.INT32)
                embeddingRows_field = trt.PluginField("embeddingRows", np.array([embeddingRows], dtype=np.int32), trt.PluginFieldType.INT32)
                embeddingWeightsOnGpuPart_field = trt.PluginField("embeddingWeightsOnGpuPart", np.array([self.embedding_weights_on_gpu_part], dtype=np.float32), trt.PluginFieldType.FLOAT32)
                tableHotness_field = trt.PluginField("tableHotness", tableHotness, trt.PluginFieldType.INT32)
                tableOffsets_field = trt.PluginField("tableOffsets", tableOffsets, trt.PluginFieldType.INT32)
                batchSize_field = trt.PluginField("batchSize", np.array([self.batch_size], dtype=np.int32), trt.PluginFieldType.INT32)
                embedHotnessTotal_field = trt.PluginField("embedHotnessTotal", np.array([totalHotness], dtype=np.int32), trt.PluginFieldType.INT32)
                embeddingWeightsFilepath_field = trt.PluginField("embeddingWeightsFilepath", np.array(list(str(self.mega_table_npy_file).encode()), dtype=np.int8), trt.PluginFieldType.CHAR)
                rowFrequenciesFilepath_field = trt.PluginField("rowFrequenciesFilepath", np.array(list(str(self.row_frequencies_npy_filepath).encode()), dtype=np.int8), trt.PluginFieldType.CHAR)
                embeddingScalesFilepath_field = trt.PluginField("embeddingScalesFilepath", np.array(list(str(self.mega_table_scales_npy_file).encode()), dtype=np.int8), trt.PluginFieldType.CHAR)
                reducedPrecisionIO_field = trt.PluginField("reducedPrecisionIO", np.array([reducedPrecisionIO], dtype=np.int32), trt.PluginFieldType.INT32)
                useRowFrequenciesOpt_field = trt.PluginField("useRowFrequenciesOpt", np.array([self.use_row_frequencies_opt], dtype=np.int32), trt.PluginFieldType.INT32)

                field_collection = trt.PluginFieldCollection([
                    embeddingSize_field,
                    embeddingRows_field,
                    embeddingWeightsOnGpuPart_field,
                    tableHotness_field,
                    tableOffsets_field,
                    batchSize_field,
                    embedHotnessTotal_field,
                    embeddingWeightsFilepath_field,
                    rowFrequenciesFilepath_field,
                    embeddingScalesFilepath_field,
                    reducedPrecisionIO_field,
                    useRowFrequenciesOpt_field
                ])
                plugin = plugin_creator.create_plugin(name=pluginName, field_collection=field_collection)

        assert plugin is not None, f"Plugin needs to be registered: {pluginName}"
        return plugin


class DLRMv2EngineBuilder(CalibratableTensorRTEngine,
                          TRTBuilder,
                          MLPerfInferenceEngine,
                          ArgDiscarder):
    def __init__(self,
                 workspace_size: int = 8 << 30,
                 config_ver: str = "default",
                 component: str = None,
                 batch_size: int = 8192,

                 # weights and embeddings paths
                 model_path: os.PathLike = Path("/home/mlperf_inf_dlrmv2/model/model_weights"),
                 embeddings_path: os.PathLike = Path('/home/mlperf_inf_dlrmv2/model/embedding_weights'),

                 # precision configs
                 input_dtype: str = 'fp16',
                 bot_mlp_precision: str = 'int8',
                 embeddings_precision: str = 'int8',
                 interaction_op_precision: str = 'int8',
                 top_mlp_precision: str = 'int8',
                 final_linear_precision: str = 'int8',

                 # feature configs
                 embedding_weights_on_gpu_part: float = 1.0,

                 # calibration setup
                 calib_batch_size: int = 256,
                 calib_max_batches: int = 500,
                 calib_data_map: os.PathLike = Path("data_maps/criteo/cal_map.txt"),
                 cache_file: os.PathLike = Path("code/dlrm-v2/tensorrt/calibrator.cache"),

                 **kwargs):
        super().__init__(workspace_size=workspace_size,
                         calib_batch_size=calib_batch_size,
                         calib_max_batches=calib_max_batches,
                         calib_data_map=calib_data_map,
                         cache_file=cache_file,
                         **kwargs)

        self.verbose = kwargs.get('verbose', False)
        self.config_ver = config_ver
        self.component = component
        self.batch_size = batch_size

        self.model_path = Path(model_path)
        self.embeddings_path = Path(embeddings_path)

        assert input_dtype in ['fp32', 'fp16', 'int8']
        assert bot_mlp_precision in ['fp32', 'fp16', 'int8']
        assert embeddings_precision in ['fp32', 'fp16', 'int8']
        assert interaction_op_precision in ['fp32', 'fp16', 'int8']
        assert top_mlp_precision in ['fp32', 'fp16', 'int8']
        assert final_linear_precision in ['fp32', 'fp16', 'int8']

        self.dense_dtype = input_dtype
        self.sparse_dtype = 'int32'
        self.bot_mlp_precision = bot_mlp_precision
        self.embeddings_precision = embeddings_precision
        self.interaction_op_precision = interaction_op_precision
        self.top_mlp_precision = top_mlp_precision
        self.final_linear_precision = final_linear_precision

        self.embedding_weights_on_gpu_part = embedding_weights_on_gpu_part
        self.use_row_frequencies_opt = self.embeddings_precision != 'int8'
        self.row_frequencies_npy_filepath = Path('/home/mlperf_inf_dlrmv2/criteo/day23/row_frequencies.npy')

        # timing cache setup
        self.use_timing_cache = True
        self.timing_cache_file = f"./build/cache/dlrm_build_cache_{self.precision}.cache"

        if self.force_calibration:
            logging.info('Building Engine in Calibration Mode')

            # NOTE(vir): fp32 everything when in calibration mode
            self.dense_dtype = 'fp32'
            self.sparse_dtype = 'int32'
            self.bot_mlp_precision = 'fp32'
            self.embeddings_precision = 'fp32'
            self.interaction_op_precision = 'fp32'
            self.top_mlp_precision = 'fp32'
            self.final_linear_precision = 'fp32'

            # NOTE(vir): ~100GB vram needed so be conservative
            self.embedding_weights_on_gpu_part = 0.3

            # NOTE(vir): WAR using row frequencies opt during calibration gives slightly different scales. Why?
            self.use_row_frequencies_opt = False

    def get_calibrator(self, _: str = None):
        return DLRMv2Calibrator(
            calib_batch_size=self.calib_batch_size,
            calib_max_batches=self.calib_max_batches,
            force_calibration=self.force_calibration,
            cache_file=self.cache_file
        )

    def create_network(self, builder: trt.Builder):
        base_network = super().create_network(builder)
        dlrm_network = DLRMv2TRTNetwork(network=base_network,
                                        batch_size=self.batch_size,
                                        verbose=self.verbose,
                                        model_path=self.model_path,
                                        embeddings_path=self.embeddings_path,
                                        dense_dtype=self.dense_dtype,
                                        sparse_dtype=self.sparse_dtype,
                                        bot_mlp_precision=self.bot_mlp_precision,
                                        embeddings_precision=self.embeddings_precision,
                                        interaction_op_precision=self.interaction_op_precision,
                                        top_mlp_precision=self.top_mlp_precision,
                                        final_linear_precision=self.final_linear_precision,
                                        embedding_weights_on_gpu_part=self.embedding_weights_on_gpu_part,
                                        use_row_frequencies_opt=self.use_row_frequencies_opt,
                                        row_frequencies_npy_filepath=self.row_frequencies_npy_filepath)

        return dlrm_network.network

    @injector.inject_dlrm_dependencies()
    def gpu_profiles(self,
                     network: trt.INetworkDefinition,
                     batch_size: int):
        min_bs = 1  # TODO(vir): this breaks cma fusion, need >= 2048
        numerical_input_profile = (batch_size, INT_FEATURE_COUNT, 1, 1)
        sparse_input_profile = (batch_size, sum(CRITEO_SYNTH_MULTIHOT_SIZES))

        profiles = [Profile()
                    .add("numerical_input", min=(min_bs, INT_FEATURE_COUNT, 1, 1), opt=numerical_input_profile, max=numerical_input_profile)
                    .add("sparse_input", min=(min_bs, sum(CRITEO_SYNTH_MULTIHOT_SIZES)), opt=sparse_input_profile, max=sparse_input_profile)
                    .to_trt(self.builder, network)]

        return profiles

    def create_builder_config(self, profiles: List[trt.IOptimizationProfile]):
        builder_config = super().create_builder_config(profiles=profiles)
        builder_config.int8_calibrator = self.get_calibrator()
        builder_config.builder_optimization_level = 4  # Needed for ConvMulAdd fusion from Myelin

        if self.use_timing_cache:
            # load existing cache if found, else create a new one
            timing_cache = b""

            if os.path.exists(self.timing_cache_file):
                with open(self.timing_cache_file, 'rb') as f:
                    timing_cache = f.read()

            trt_timing_cache = builder_config.create_timing_cache(timing_cache)
            builder_config.set_timing_cache(trt_timing_cache, False)
            logging.info(f'Using Timing Cache: {self.timing_cache_file}')

        builder_config.set_flag(trt.BuilderFlag.FP16)
        builder_config.set_flag(trt.BuilderFlag.INT8)
        builder_config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        return builder_config


class DLRMv2EngineBuilderOp(Operation,
                            ArgDiscarder):
    COMPONENT_BUILDER_MAP = {
        DLRMv2Component.DLRMv2: DLRMv2EngineBuilder,
    }

    @classmethod
    def immediate_dependencies(cls):
        return None

    def __init__(self,
                 *args,
                 # Benchmark specific values
                 batch_size: Dict[DLRMv2Component, int] = None,
                 **kwargs):
        """Creates a DLRMv2EngineBuilderOp.

        Args:
            batch_size (Dict[str, int]): Component and its batch size to build the engine for)
        """
        super().__init__(*args, **kwargs)
        if not batch_size:
            logging.warning("No batch_size dict provided for DLRMv2EngineBuilderOp. Setting to default value {DLRMv2Component.DLRMv2 : 1}")
            batch_size = {DLRMv2Component.DLRMv2: 1}
        self.builders = []
        for component, component_batch_size in batch_size.items():
            builder = DLRMv2EngineBuilderOp.COMPONENT_BUILDER_MAP[component](*args, component=component.valstr(), batch_size=component_batch_size, **kwargs)
            self.builders.append(builder)

    def run(self, scratch_space, dependency_outputs):
        for builder in self.builders:
            network = builder.create_network(builder.builder)
            profiles = builder.create_profiles(network, builder.batch_size)
            builder_config = builder.create_builder_config(profiles)
            engine_dir = builder.engine_dir(scratch_space)
            engine_name = builder.engine_name("gpu",
                                              builder.batch_size,
                                              builder.precision,
                                              builder.component,
                                              tag=builder.config_ver)
            engine_fpath = engine_dir / engine_name

            builder(builder.batch_size, engine_fpath, network, profiles, builder_config)

            if builder.use_timing_cache:
                Path(builder.timing_cache_file).parent.mkdir(parents=True, exist_ok=True)

                # save latest timing cache
                with open(builder.timing_cache_file, 'wb') as f:
                    f.write(builder_config.get_timing_cache().serialize())

                logging.info(f'Timing Cache Updated: {builder.timing_cache_file}')


class DLRMv2(LegacyBuilder):
    """Temporary spoofing class to wrap around Mitten to adhere to the old API. """

    def __init__(self, args):
        self.mitten_builder = DLRMv2EngineBuilderOp(**args)
        super().__init__(self.mitten_builder)

    @injector.inject_dlrm_dependencies()
    def generate_engines(self, *args, **kwargs):
        return super().generate_engines(*args, **kwargs)

    @injector.inject_dlrm_dependencies()
    def calibrate(self):
        # NOTE(VIR): WAR: override mitten calibration with polygraphy, which gives better scales

        # calibration data params
        lower_bound = 89137319
        upper_bound = 89265318
        batch_size = 256
        num_batches = 1 + ((upper_bound - lower_bound) // batch_size)

        # load dataset
        dataset = CriteoDay23Dataset()

        # create network
        @func.extend(CreateNetwork())
        def create_network(_, network):
            DLRMv2TRTNetwork(network=network,
                             batch_size=batch_size,
                             verbose=self.mitten_builder.verbose,
                             model_path=self.mitten_builder.model_path,
                             embeddings_path=self.mitten_builder.embeddings_path,
                             dense_dtype=self.mitten_builder.dense_dtype,
                             sparse_dtype=self.mitten_builder.sparse_dtype,
                             bot_mlp_precision=self.mitten_builder.bot_mlp_precision,
                             embeddings_precision=self.mitten_builder.embeddings_precision,
                             interaction_op_precision=self.mitten_builder.interaction_op_precision,
                             top_mlp_precision=self.mitten_builder.top_mlp_precision,
                             final_linear_precision=self.mitten_builder.final_linear_precision,
                             embedding_weights_on_gpu_part=self.mitten_builder.embedding_weights_on_gpu_part,
                             use_row_frequencies_opt=self.mitten_builder.use_row_frequencies_opt,
                             row_frequencies_npy_filepath=self.mitten_builder.row_frequencies_npy_filepath)

        # calibration dataset generator
        @injector.inject_dlrm_dependencies()
        def data_loader():
            for idx in range(num_batches):
                s = lower_bound + (idx + 0) * batch_size
                e = lower_bound + (idx + 1) * batch_size
                assert s < upper_bound and e <= upper_bound + 1

                batch = dataset.get_batch(indices=np.arange(s, e))
                dense_input = np.ascontiguousarray(batch["dense"], dtype=np.float32).reshape(batch_size, INT_FEATURE_COUNT, 1, 1)
                sparse_input = np.ascontiguousarray(np.hstack(batch["sparse"]), dtype=np.int32).reshape(batch_size, sum(CRITEO_SYNTH_MULTIHOT_SIZES))

                yield {'numerical_input': dense_input, 'sparse_input': sparse_input}

        profiles = [Profile()
                    .add("numerical_input", min=(batch_size, INT_FEATURE_COUNT, 1, 1), opt=(batch_size, INT_FEATURE_COUNT, 1, 1), max=(batch_size, INT_FEATURE_COUNT, 1, 1))
                    .add("sparse_input", min=(batch_size, sum(CRITEO_SYNTH_MULTIHOT_SIZES)), opt=(batch_size, sum(CRITEO_SYNTH_MULTIHOT_SIZES)), max=(batch_size, sum(CRITEO_SYNTH_MULTIHOT_SIZES)))]
        calibrator = Calibrator(data_loader=data_loader(), cache=self.mitten_builder.cache_file, batch_size=batch_size)
        load_engine = EngineFromNetwork(create_network, config=CreateConfig(int8=True, calibrator=calibrator, profiles=profiles, precision_constraints='obey'))

        # run calibration
        with G_LOGGER.verbosity(G_LOGGER.VERBOSE), TrtRunner(load_engine) as _:
            logging.info(f'Calibration completed, cache written to: {self.mitten_builder.cache_file}')
            pass
