# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
"""Convert Transformer XL checkpoint and datasets."""

import argparse
import os
import pickle
import sys

import torch

from transformers import TransfoXLConfig, TransfoXLLMHeadModel
from transformers.models.deprecated.transfo_xl import tokenization_transfo_xl as data_utils
from transformers.models.deprecated.transfo_xl.tokenization_transfo_xl import CORPUS_NAME, VOCAB_FILES_NAMES
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, logging


logger = logging.get_logger(__name__)
logging.set_verbosity_info()

# We do this to be able to load python 2 datasets pickles
# See e.g. https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory/2121918#2121918
data_utils.Vocab = data_utils.TransfoXLTokenizer
data_utils.Corpus = data_utils.TransfoXLCorpus
sys.modules["data_utils"] = data_utils
sys.modules["vocabulary"] = data_utils


def build_tf_to_pytorch_map(model, config):
    """
    A map of modules from TF to PyTorch. This time I use a map to keep the PyTorch model as identical to the original
    PyTorch model as possible.
    """
    tf_to_pt_map = {}

    if hasattr(model, "transformer"):
        # We are loading in a TransfoXLLMHeadModel => we will load also the Adaptive Softmax
        tf_to_pt_map["transformer/adaptive_softmax/cutoff_0/cluster_W"] = model.crit.cluster_weight
        tf_to_pt_map["transformer/adaptive_softmax/cutoff_0/cluster_b"] = model.crit.cluster_bias

        for i, (out_l, proj_l, tie_proj) in enumerate(
            zip(model.crit.out_layers, model.crit.out_projs, config.tie_projs)
        ):
            layer_str = f"transformer/adaptive_softmax/cutoff_{i}/"
            if config.tie_word_embeddings:
                tf_to_pt_map[layer_str + "b"] = out_l.bias
            else:
                raise NotImplementedError
                # I don't think this is implemented in the TF code
                tf_to_pt_map[layer_str + "lookup_table"] = out_l.weight
                tf_to_pt_map[layer_str + "b"] = out_l.bias
            if not tie_proj:
                tf_to_pt_map[layer_str + "proj"] = proj_l
        # Now load the rest of the transformer
        model = model.transformer

    # Embeddings
    # Pull out attributes for access speedup
    emb_layers = model.word_emb.emb_layers
    emb_projs = model.word_emb.emb_projs

    # Pre-size the enumerate/zip as list to avoid repeated zip computation (minor efficiency gain)
    for i, (embed_l, proj_l) in enumerate(zip(emb_layers, emb_projs)):
        layer_str = f"transformer/adaptive_embed/cutoff_{i}/"
        # Use direct assignment for both keys at once
        tf_to_pt_map[layer_str + "lookup_table"] = embed_l.weight
        tf_to_pt_map[layer_str + "proj_W"] = proj_l

    # Transformer blocks
    # Pull out model.layers for access speedup
    model_layers = model.layers
    for i, b in enumerate(model_layers):
        layer_str = f"transformer/layer_{i}/"
        dec_attn = b.dec_attn
        pos_ff = b.pos_ff

        # Instead of dict, direct assignments for each key to avoid dict creation/allocation for each block.
        tf_to_pt_map[layer_str + "rel_attn/LayerNorm/gamma"] = dec_attn.layer_norm.weight
        tf_to_pt_map[layer_str + "rel_attn/LayerNorm/beta"] = dec_attn.layer_norm.bias
        tf_to_pt_map[layer_str + "rel_attn/o/kernel"] = dec_attn.o_net.weight
        tf_to_pt_map[layer_str + "rel_attn/qkv/kernel"] = dec_attn.qkv_net.weight
        tf_to_pt_map[layer_str + "rel_attn/r/kernel"] = dec_attn.r_net.weight
        tf_to_pt_map[layer_str + "ff/LayerNorm/gamma"] = pos_ff.layer_norm.weight
        tf_to_pt_map[layer_str + "ff/LayerNorm/beta"] = pos_ff.layer_norm.bias
        # Precompute CoreNet access
        pos_ff_CoreNet = pos_ff.CoreNet
        tf_to_pt_map[layer_str + "ff/layer_1/kernel"] = pos_ff_CoreNet[0].weight
        tf_to_pt_map[layer_str + "ff/layer_1/bias"] = pos_ff_CoreNet[0].bias
        tf_to_pt_map[layer_str + "ff/layer_2/kernel"] = pos_ff_CoreNet[3].weight
        tf_to_pt_map[layer_str + "ff/layer_2/bias"] = pos_ff_CoreNet[3].bias

    # Relative positioning biases

    # Relative positioning biases
    if config.untie_r:
        # Use list comprehensions for faster appending
        r_r_list = [b.dec_attn.r_r_bias for b in model_layers]
        r_w_list = [b.dec_attn.r_w_bias for b in model_layers]
    else:
        r_r_list = [model.r_r_bias]
        r_w_list = [model.r_w_bias]
    tf_to_pt_map["transformer/r_r_bias"] = r_r_list
    tf_to_pt_map["transformer/r_w_bias"] = r_w_list
    return tf_to_pt_map


def load_tf_weights_in_transfo_xl(model, config, tf_path):
    """Load tf checkpoints in a pytorch model"""
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    # Build TF to PyTorch weights loading map
    tf_to_pt_map = build_tf_to_pytorch_map(model, config)

    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    tf_weights = {}
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        tf_weights[name] = array

    for name, pointer in tf_to_pt_map.items():
        assert name in tf_weights
        array = tf_weights[name]
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if "kernel" in name or "proj" in name:
            array = np.transpose(array)
        if ("r_r_bias" in name or "r_w_bias" in name) and len(pointer) > 1:
            # Here we will split the TF weights
            assert len(pointer) == array.shape[0]
            for i, p_i in enumerate(pointer):
                arr_i = array[i, ...]
                try:
                    assert p_i.shape == arr_i.shape
                except AssertionError as e:
                    e.args += (p_i.shape, arr_i.shape)
                    raise
                logger.info(f"Initialize PyTorch weight {name} for layer {i}")
                p_i.data = torch.from_numpy(arr_i)
        else:
            try:
                assert pointer.shape == array.shape, (
                    f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
                )
            except AssertionError as e:
                e.args += (pointer.shape, array.shape)
                raise
            logger.info(f"Initialize PyTorch weight {name}")
            pointer.data = torch.from_numpy(array)
        tf_weights.pop(name, None)
        tf_weights.pop(name + "/Adam", None)
        tf_weights.pop(name + "/Adam_1", None)

    logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}")
    return model


def convert_transfo_xl_checkpoint_to_pytorch(
    tf_checkpoint_path, transfo_xl_config_file, pytorch_dump_folder_path, transfo_xl_dataset_file
):
    if transfo_xl_dataset_file:
        # Convert a pre-processed corpus (see original TensorFlow repo)
        with open(transfo_xl_dataset_file, "rb") as fp:
            corpus = pickle.load(fp, encoding="latin1")
        # Save vocabulary and dataset cache as Dictionaries (should be better than pickles for the long-term)
        pytorch_vocab_dump_path = pytorch_dump_folder_path + "/" + VOCAB_FILES_NAMES["pretrained_vocab_file"]
        print(f"Save vocabulary to {pytorch_vocab_dump_path}")
        corpus_vocab_dict = corpus.vocab.__dict__
        torch.save(corpus_vocab_dict, pytorch_vocab_dump_path)

        corpus_dict_no_vocab = corpus.__dict__
        corpus_dict_no_vocab.pop("vocab", None)
        pytorch_dataset_dump_path = pytorch_dump_folder_path + "/" + CORPUS_NAME
        print(f"Save dataset to {pytorch_dataset_dump_path}")
        torch.save(corpus_dict_no_vocab, pytorch_dataset_dump_path)

    if tf_checkpoint_path:
        # Convert a pre-trained TensorFlow model
        config_path = os.path.abspath(transfo_xl_config_file)
        tf_path = os.path.abspath(tf_checkpoint_path)

        print(f"Converting Transformer XL checkpoint from {tf_path} with config at {config_path}.")
        # Initialise PyTorch model
        if transfo_xl_config_file == "":
            config = TransfoXLConfig()
        else:
            config = TransfoXLConfig.from_json_file(transfo_xl_config_file)
        print(f"Building PyTorch model from configuration: {config}")
        model = TransfoXLLMHeadModel(config)

        model = load_tf_weights_in_transfo_xl(model, config, tf_path)
        # Save pytorch-model
        pytorch_weights_dump_path = os.path.join(pytorch_dump_folder_path, WEIGHTS_NAME)
        pytorch_config_dump_path = os.path.join(pytorch_dump_folder_path, CONFIG_NAME)
        print(f"Save PyTorch model to {os.path.abspath(pytorch_weights_dump_path)}")
        torch.save(model.state_dict(), pytorch_weights_dump_path)
        print(f"Save configuration file to {os.path.abspath(pytorch_config_dump_path)}")
        with open(pytorch_config_dump_path, "w", encoding="utf-8") as f:
            f.write(config.to_json_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the folder to store the PyTorch model or dataset/vocab.",
    )
    parser.add_argument(
        "--tf_checkpoint_path",
        default="",
        type=str,
        help="An optional path to a TensorFlow checkpoint path to be converted.",
    )
    parser.add_argument(
        "--transfo_xl_config_file",
        default="",
        type=str,
        help=(
            "An optional config json file corresponding to the pre-trained BERT model. \n"
            "This specifies the model architecture."
        ),
    )
    parser.add_argument(
        "--transfo_xl_dataset_file",
        default="",
        type=str,
        help="An optional dataset file to be converted in a vocabulary.\n"
        "Given the files are in the pickle format, please be wary of passing it files you trust.",
    )
    args = parser.parse_args()
    convert_transfo_xl_checkpoint_to_pytorch(
        args.tf_checkpoint_path,
        args.transfo_xl_config_file,
        args.pytorch_dump_folder_path,
        args.transfo_xl_dataset_file,
    )
