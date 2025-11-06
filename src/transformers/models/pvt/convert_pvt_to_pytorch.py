# coding=utf-8
# Copyright 2023 Authors: Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan,
# Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao and The HuggingFace Inc. team.
# All rights reserved.
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
"""Convert Pvt checkpoints from the original library."""

import argparse
from pathlib import Path

import requests
import torch
from PIL import Image

from transformers import PvtConfig, PvtForImageClassification, PvtImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config):
    # Preallocate list size using maximum length estimation, if possible.
    # Otherwise, minimize repeated append overhead by using local vars.
    depths = config.depths
    num_encoder_blocks = config.num_encoder_blocks
    sequence_reduction_ratios = config.sequence_reduction_ratios

    # Because 8 keys before the innermost loop and up to 16-24 per inner block depending on ratio.
    precompute_sz = (num_encoder_blocks * (5 + sum(depths) * 15)) + 7
    rename_keys = []
    append = rename_keys.append  # Local function var for faster access

    for i in range(num_encoder_blocks):
        idx1 = i + 1
        # Rename embeddings' parameters
        append((f"pos_embed{idx1}", f"pvt.encoder.patch_embeddings.{i}.position_embeddings"))
        append((f"patch_embed{idx1}.proj.weight", f"pvt.encoder.patch_embeddings.{i}.projection.weight"))
        append((f"patch_embed{idx1}.proj.bias", f"pvt.encoder.patch_embeddings.{i}.projection.bias"))
        append((f"patch_embed{idx1}.norm.weight", f"pvt.encoder.patch_embeddings.{i}.layer_norm.weight"))
        append((f"patch_embed{idx1}.norm.bias", f"pvt.encoder.patch_embeddings.{i}.layer_norm.bias"))

        depth = depths[i]
        sr_ratio = sequence_reduction_ratios[i]
        for j in range(depth):
            # Prepare strings only once per pattern for this block/j, then interpolate just the indices
            idx_j = f"{i}.{j}"
            block_prefix = f"block{idx1}.{j}"
            encoder_block_prefix = f"pvt.encoder.block.{i}.{j}"

            # Rename blocks' parameters
            append(
                (f"{block_prefix}.attn.q.weight", f"{encoder_block_prefix}.attention.self.query.weight")
            )
            append(
                (f"{block_prefix}.attn.q.bias", f"{encoder_block_prefix}.attention.self.query.bias")
            )
            append(
                (f"{block_prefix}.attn.kv.weight", f"{encoder_block_prefix}.attention.self.kv.weight")
            )
            append(
                (f"{block_prefix}.attn.kv.bias", f"{encoder_block_prefix}.attention.self.kv.bias")
            )

            if sr_ratio > 1:
                append(
                    (
                        f"{block_prefix}.attn.norm.weight",
                        f"{encoder_block_prefix}.attention.self.layer_norm.weight",
                    )
                )
                append(
                    (
                        f"{block_prefix}.attn.norm.bias",
                        f"{encoder_block_prefix}.attention.self.layer_norm.bias",
                    )
                )
                append(
                    (
                        f"{block_prefix}.attn.sr.weight",
                        f"{encoder_block_prefix}.attention.self.sequence_reduction.weight",
                    )
                )
                append(
                    (
                        f"{block_prefix}.attn.sr.bias",
                        f"{encoder_block_prefix}.attention.self.sequence_reduction.bias",
                    )
                )

            append(
                (f"{block_prefix}.attn.proj.weight", f"{encoder_block_prefix}.attention.output.dense.weight")
            )
            append(
                (f"{block_prefix}.attn.proj.bias", f"{encoder_block_prefix}.attention.output.dense.bias")
            )

            append(
                (f"{block_prefix}.norm1.weight", f"{encoder_block_prefix}.layer_norm_1.weight")
            )
            append(
                (f"{block_prefix}.norm1.bias", f"{encoder_block_prefix}.layer_norm_1.bias")
            )
            append(
                (f"{block_prefix}.norm2.weight", f"{encoder_block_prefix}.layer_norm_2.weight")
            )
            append(
                (f"{block_prefix}.norm2.bias", f"{encoder_block_prefix}.layer_norm_2.bias")
            )

            append(
                (f"{block_prefix}.mlp.fc1.weight", f"{encoder_block_prefix}.mlp.dense1.weight")
            )
            append(
                (f"{block_prefix}.mlp.fc1.bias", f"{encoder_block_prefix}.mlp.dense1.bias")
            )
            append(
                (f"{block_prefix}.mlp.fc2.weight", f"{encoder_block_prefix}.mlp.dense2.weight")
            )
            append(
                (f"{block_prefix}.mlp.fc2.bias", f"{encoder_block_prefix}.mlp.dense2.bias")
            )

    # Rename cls token

    # Rename cls token
    rename_keys.extend(
        [
            ("cls_token", "pvt.encoder.patch_embeddings.3.cls_token"),
        ]
    )
    # Rename norm layer and classifier layer
    rename_keys.extend(
        [
            ("norm.weight", "pvt.encoder.layer_norm.weight"),
            ("norm.bias", "pvt.encoder.layer_norm.bias"),
            ("head.weight", "classifier.weight"),
            ("head.bias", "classifier.bias"),
        ]
    )

    return rename_keys


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_k_v(state_dict, config):
    # for each of the encoder blocks:
    for i in range(config.num_encoder_blocks):
        for j in range(config.depths[i]):
            # read in weights + bias of keys and values (which is a single matrix in the original implementation)
            kv_weight = state_dict.pop(f"pvt.encoder.block.{i}.{j}.attention.self.kv.weight")
            kv_bias = state_dict.pop(f"pvt.encoder.block.{i}.{j}.attention.self.kv.bias")
            # next, add keys and values (in that order) to the state dict
            state_dict[f"pvt.encoder.block.{i}.{j}.attention.self.key.weight"] = kv_weight[: config.hidden_sizes[i], :]
            state_dict[f"pvt.encoder.block.{i}.{j}.attention.self.key.bias"] = kv_bias[: config.hidden_sizes[i]]

            state_dict[f"pvt.encoder.block.{i}.{j}.attention.self.value.weight"] = kv_weight[
                config.hidden_sizes[i] :, :
            ]
            state_dict[f"pvt.encoder.block.{i}.{j}.attention.self.value.bias"] = kv_bias[config.hidden_sizes[i] :]


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_pvt_checkpoint(pvt_size, pvt_checkpoint, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our PVT structure.
    """

    # define default Pvt configuration
    if pvt_size == "tiny":
        config_path = "Zetatech/pvt-tiny-224"
    elif pvt_size == "small":
        config_path = "Zetatech/pvt-small-224"
    elif pvt_size == "medium":
        config_path = "Zetatech/pvt-medium-224"
    elif pvt_size == "large":
        config_path = "Zetatech/pvt-large-224"
    else:
        raise ValueError(f"Available model's size: 'tiny', 'small', 'medium', 'large', but '{pvt_size}' was given")
    config = PvtConfig(name_or_path=config_path)
    # load original model from https://github.com/whai362/PVT
    state_dict = torch.load(pvt_checkpoint, map_location="cpu", weights_only=True)

    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_k_v(state_dict, config)

    # load HuggingFace model
    model = PvtForImageClassification(config).eval()
    model.load_state_dict(state_dict)

    # Check outputs on an image, prepared by PVTFeatureExtractor
    image_processor = PvtImageProcessor(size=config.image_size)
    encoding = image_processor(images=prepare_img(), return_tensors="pt")
    pixel_values = encoding["pixel_values"]
    outputs = model(pixel_values)
    logits = outputs.logits.detach().cpu()

    if pvt_size == "tiny":
        expected_slice_logits = torch.tensor([-1.4192, -1.9158, -0.9702])
    elif pvt_size == "small":
        expected_slice_logits = torch.tensor([0.4353, -0.1960, -0.2373])
    elif pvt_size == "medium":
        expected_slice_logits = torch.tensor([-0.2914, -0.2231, 0.0321])
    elif pvt_size == "large":
        expected_slice_logits = torch.tensor([0.3740, -0.7739, -0.4214])
    else:
        raise ValueError(f"Available model's size: 'tiny', 'small', 'medium', 'large', but '{pvt_size}' was given")

    assert torch.allclose(logits[0, :3], expected_slice_logits, atol=1e-4)

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model pytorch_model.bin to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    image_processor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--pvt_size",
        default="tiny",
        type=str,
        help="Size of the PVT pretrained model you'd like to convert.",
    )
    parser.add_argument(
        "--pvt_checkpoint",
        default="pvt_tiny.pth",
        type=str,
        help="Checkpoint of the PVT pretrained model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_pvt_checkpoint(args.pvt_size, args.pvt_checkpoint, args.pytorch_dump_folder_path)
