# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""Convert RT Detr checkpoints with Timm backbone"""

import argparse
import json
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from transformers import RTDetrConfig, RTDetrForObjectDetection, RTDetrImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_rt_detr_config(model_name: str) -> RTDetrConfig:
    config = RTDetrConfig()

    config.num_labels = 80
    repo_id = "huggingface/label-files"
    filename = "coco-detection-mmdet-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    if model_name == "rtdetr_r18vd":
        config.backbone_config.hidden_sizes = [64, 128, 256, 512]
        config.backbone_config.depths = [2, 2, 2, 2]
        config.backbone_config.layer_type = "basic"
        config.encoder_in_channels = [128, 256, 512]
        config.hidden_expansion = 0.5
        config.decoder_layers = 3
    elif model_name == "rtdetr_r34vd":
        config.backbone_config.hidden_sizes = [64, 128, 256, 512]
        config.backbone_config.depths = [3, 4, 6, 3]
        config.backbone_config.layer_type = "basic"
        config.encoder_in_channels = [128, 256, 512]
        config.hidden_expansion = 0.5
        config.decoder_layers = 4
    elif model_name == "rtdetr_r50vd_m":
        pass
    elif model_name == "rtdetr_r50vd":
        pass
    elif model_name == "rtdetr_r101vd":
        config.backbone_config.depths = [3, 4, 23, 3]
        config.encoder_ffn_dim = 2048
        config.encoder_hidden_dim = 384
        config.decoder_in_channels = [384, 384, 384]
    elif model_name == "rtdetr_r18vd_coco_o365":
        config.backbone_config.hidden_sizes = [64, 128, 256, 512]
        config.backbone_config.depths = [2, 2, 2, 2]
        config.backbone_config.layer_type = "basic"
        config.encoder_in_channels = [128, 256, 512]
        config.hidden_expansion = 0.5
        config.decoder_layers = 3
    elif model_name == "rtdetr_r50vd_coco_o365":
        pass
    elif model_name == "rtdetr_r101vd_coco_o365":
        config.backbone_config.depths = [3, 4, 23, 3]
        config.encoder_ffn_dim = 2048
        config.encoder_hidden_dim = 384
        config.decoder_in_channels = [384, 384, 384]

    return config


def create_rename_keys(config):
    # here we list all keys to be renamed (original name on the left, our name on the right)
    rename_keys = []
    last_key = ["weight", "bias", "running_mean", "running_var"]

    # Precompute repeated values for layer_type and depth-related ranges
    backbone_depths = config.backbone_config.depths
    layer_type = config.backbone_config.layer_type
    encoder_layers = config.encoder_layers
    decoder_layers = config.decoder_layers
    encoder_in_channels = config.encoder_in_channels
    decoder_in_channels = config.decoder_in_channels

    block_levels = 3 if layer_type != "basic" else 4
    len_encoder_in_channels = len(encoder_in_channels)
    len_decoder_in_channels = len(decoder_in_channels)

    append = rename_keys.append  # Localize method for speed

    # stem
    for level in range(3):
        append((
            f"backbone.conv1.conv1_{level+1}.conv.weight",
            f"model.backbone.model.embedder.embedder.{level}.convolution.weight",
        ))
        norm_pre = f"backbone.conv1.conv1_{level+1}.norm."
        norm_post = f"model.backbone.model.embedder.embedder.{level}.normalization."
        # Unroll to avoid Python loop overhead for short, fixed-length sequences
        append((f"{norm_pre}weight", f"{norm_post}weight"))
        append((f"{norm_pre}bias", f"{norm_post}bias"))
        append((f"{norm_pre}running_mean", f"{norm_post}running_mean"))
        append((f"{norm_pre}running_var", f"{norm_post}running_var"))

    # backbone.res_layers and friends
    for stage_idx, stage_depth in enumerate(backbone_depths):
        for layer_idx in range(stage_depth):
            # shortcut
            if layer_idx == 0:
                if stage_idx == 0:
                    append((
                        f"backbone.res_layers.{stage_idx}.blocks.0.short.conv.weight",
                        f"model.backbone.model.encoder.stages.{stage_idx}.layers.0.shortcut.convolution.weight",
                    ))
                    norm_pre = f"backbone.res_layers.{stage_idx}.blocks.0.short.norm."
                    norm_post = f"model.backbone.model.encoder.stages.{stage_idx}.layers.0.shortcut.normalization."
                    append((f"{norm_pre}weight", f"{norm_post}weight"))
                    append((f"{norm_pre}bias", f"{norm_post}bias"))
                    append((f"{norm_pre}running_mean", f"{norm_post}running_mean"))
                    append((f"{norm_pre}running_var", f"{norm_post}running_var"))
                else:
                    append((
                        f"backbone.res_layers.{stage_idx}.blocks.0.short.conv.conv.weight",
                        f"model.backbone.model.encoder.stages.{stage_idx}.layers.0.shortcut.1.convolution.weight",
                    ))
                    norm_pre = f"backbone.res_layers.{stage_idx}.blocks.0.short.conv.norm."
                    norm_post = f"model.backbone.model.encoder.stages.{stage_idx}.layers.0.shortcut.1.normalization."
                    append((f"{norm_pre}weight", f"{norm_post}weight"))
                    append((f"{norm_pre}bias", f"{norm_post}bias"))
                    append((f"{norm_pre}running_mean", f"{norm_post}running_mean"))
                    append((f"{norm_pre}running_var", f"{norm_post}running_var"))

            idx_prefix = f"backbone.res_layers.{stage_idx}.blocks.{layer_idx}"
            idx_postfix0 = f"model.backbone.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.0"
            append((f"{idx_prefix}.branch2a.conv.weight", f"{idx_postfix0}.convolution.weight"))
            norm_pre = f"{idx_prefix}.branch2a.norm."
            norm_post = f"{idx_postfix0}.normalization."
            append((f"{norm_pre}weight", f"{norm_post}weight"))
            append((f"{norm_pre}bias", f"{norm_post}bias"))
            append((f"{norm_pre}running_mean", f"{norm_post}running_mean"))
            append((f"{norm_pre}running_var", f"{norm_post}running_var"))

            idx_postfix1 = f"model.backbone.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.1"
            append((f"{idx_prefix}.branch2b.conv.weight", f"{idx_postfix1}.convolution.weight"))
            norm_pre = f"{idx_prefix}.branch2b.norm."
            norm_post = f"{idx_postfix1}.normalization."
            append((f"{norm_pre}weight", f"{norm_post}weight"))
            append((f"{norm_pre}bias", f"{norm_post}bias"))
            append((f"{norm_pre}running_mean", f"{norm_post}running_mean"))
            append((f"{norm_pre}running_var", f"{norm_post}running_var"))

            if layer_type != "basic":
                idx_postfix2 = f"model.backbone.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.2"
                append((f"{idx_prefix}.branch2c.conv.weight", f"{idx_postfix2}.convolution.weight"))
                norm_pre = f"{idx_prefix}.branch2c.norm."
                norm_post = f"{idx_postfix2}.normalization."
                append((f"{norm_pre}weight", f"{norm_post}weight"))
                append((f"{norm_pre}bias", f"{norm_post}bias"))
                append((f"{norm_pre}running_mean", f"{norm_post}running_mean"))
                append((f"{norm_pre}running_var", f"{norm_post}running_var"))

    # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
    for i in range(encoder_layers):
        enc_post = f"model.encoder.encoder.{i}.layers.0"
        enc_pre = f"encoder.encoder.{i}.layers.0"
        append((f"{enc_pre}.self_attn.out_proj.weight", f"{enc_post}.self_attn.out_proj.weight"))
        append((f"{enc_pre}.self_attn.out_proj.bias", f"{enc_post}.self_attn.out_proj.bias"))
        append((f"{enc_pre}.linear1.weight", f"{enc_post}.fc1.weight"))
        append((f"{enc_pre}.linear1.bias", f"{enc_post}.fc1.bias"))
        append((f"{enc_pre}.linear2.weight", f"{enc_post}.fc2.weight"))
        append((f"{enc_pre}.linear2.bias", f"{enc_post}.fc2.bias"))
        append((f"{enc_pre}.norm1.weight", f"{enc_post}.self_attn_layer_norm.weight"))
        append((f"{enc_pre}.norm1.bias", f"{enc_post}.self_attn_layer_norm.bias"))
        append((f"{enc_pre}.norm2.weight", f"{enc_post}.final_layer_norm.weight"))
        append((f"{enc_pre}.norm2.bias", f"{enc_post}.final_layer_norm.bias"))

    # encoder input proj
    for j in range(3):
        append((f"encoder.input_proj.{j}.0.weight", f"model.encoder_input_proj.{j}.0.weight"))
        norm_pre = f"encoder.input_proj.{j}.1."
        norm_post = f"model.encoder_input_proj.{j}.1."
        append((f"{norm_pre}weight", f"{norm_post}weight"))
        append((f"{norm_pre}bias", f"{norm_post}bias"))
        append((f"{norm_pre}running_mean", f"{norm_post}running_mean"))
        append((f"{norm_pre}running_var", f"{norm_post}running_var"))

    # block levels for fpn, lateral, bottlenecks, pan
    for i in range(len_encoder_in_channels - 1):
        # FPN blocks
        for j in range(1, block_levels):
            conv_pre = f"encoder.fpn_blocks.{i}.conv{j}."
            conv_post = f"model.encoder.fpn_blocks.{i}.conv{j}."
            append((f"{conv_pre}conv.weight", f"{conv_post}conv.weight"))
            norm_pre = f"{conv_pre}norm."
            norm_post = f"{conv_post}norm."
            append((f"{norm_pre}weight", f"{norm_post}weight"))
            append((f"{norm_pre}bias", f"{norm_post}bias"))
            append((f"{norm_pre}running_mean", f"{norm_post}running_mean"))
            append((f"{norm_pre}running_var", f"{norm_post}running_var"))

        # lateral convs
        append((f"encoder.lateral_convs.{i}.conv.weight", f"model.encoder.lateral_convs.{i}.conv.weight"))
        norm_pre = f"encoder.lateral_convs.{i}.norm."
        norm_post = f"model.encoder.lateral_convs.{i}.norm."
        append((f"{norm_pre}weight", f"{norm_post}weight"))
        append((f"{norm_pre}bias", f"{norm_post}bias"))
        append((f"{norm_pre}running_mean", f"{norm_post}running_mean"))
        append((f"{norm_pre}running_var", f"{norm_post}running_var"))

        # FPN bottlenecks
        for j in range(3):
            for k in range(1, 3):
                b_pre = f"encoder.fpn_blocks.{i}.bottlenecks.{j}.conv{k}."
                b_post = f"model.encoder.fpn_blocks.{i}.bottlenecks.{j}.conv{k}."
                append((f"{b_pre}conv.weight", f"{b_post}conv.weight"))
                norm_pre = f"{b_pre}norm."
                norm_post = f"{b_post}norm."
                append((f"{norm_pre}weight", f"{norm_post}weight"))
                append((f"{norm_pre}bias", f"{norm_post}bias"))
                append((f"{norm_pre}running_mean", f"{norm_post}running_mean"))
                append((f"{norm_pre}running_var", f"{norm_post}running_var"))

        # pan blocks
        for j in range(1, block_levels):
            conv_pre = f"encoder.pan_blocks.{i}.conv{j}."
            conv_post = f"model.encoder.pan_blocks.{i}.conv{j}."
            append((f"{conv_pre}conv.weight", f"{conv_post}conv.weight"))
            norm_pre = f"{conv_pre}norm."
            norm_post = f"{conv_post}norm."
            append((f"{norm_pre}weight", f"{norm_post}weight"))
            append((f"{norm_pre}bias", f"{norm_post}bias"))
            append((f"{norm_pre}running_mean", f"{norm_post}running_mean"))
            append((f"{norm_pre}running_var", f"{norm_post}running_var"))

        # PAN bottlenecks
        for j in range(3):
            for k in range(1, 3):
                b_pre = f"encoder.pan_blocks.{i}.bottlenecks.{j}.conv{k}."
                b_post = f"model.encoder.pan_blocks.{i}.bottlenecks.{j}.conv{k}."
                append((f"{b_pre}conv.weight", f"{b_post}conv.weight"))
                norm_pre = f"{b_pre}norm."
                norm_post = f"{b_post}norm."
                append((f"{norm_pre}weight", f"{norm_post}weight"))
                append((f"{norm_pre}bias", f"{norm_post}bias"))
                append((f"{norm_pre}running_mean", f"{norm_post}running_mean"))
                append((f"{norm_pre}running_var", f"{norm_post}running_var"))

        # Downsample convs
        append((f"encoder.downsample_convs.{i}.conv.weight", f"model.encoder.downsample_convs.{i}.conv.weight"))
        norm_pre = f"encoder.downsample_convs.{i}.norm."
        norm_post = f"model.encoder.downsample_convs.{i}.norm."
        append((f"{norm_pre}weight", f"{norm_post}weight"))
        append((f"{norm_pre}bias", f"{norm_post}bias"))
        append((f"{norm_pre}running_mean", f"{norm_post}running_mean"))
        append((f"{norm_pre}running_var", f"{norm_post}running_var"))

    # decoder layers (attention, ff, ln, heads, ...). Kept as-is to preserve semantic clarity, but slight loop unrolling for efficiency
    for i in range(decoder_layers):
        dec_post = f"model.decoder.layers.{i}"
        dec_pre = f"decoder.decoder.layers.{i}"
        append((f"{dec_pre}.self_attn.out_proj.weight", f"{dec_post}.self_attn.out_proj.weight"))
        append((f"{dec_pre}.self_attn.out_proj.bias", f"{dec_post}.self_attn.out_proj.bias"))
        ca_pre = f"{dec_pre}.cross_attn."
        ca_post = f"{dec_post}.encoder_attn."
        append((f"{ca_pre}sampling_offsets.weight", f"{ca_post}sampling_offsets.weight"))
        append((f"{ca_pre}sampling_offsets.bias", f"{ca_post}sampling_offsets.bias"))
        append((f"{ca_pre}attention_weights.weight", f"{ca_post}attention_weights.weight"))
        append((f"{ca_pre}attention_weights.bias", f"{ca_post}attention_weights.bias"))
        append((f"{ca_pre}value_proj.weight", f"{ca_post}value_proj.weight"))
        append((f"{ca_pre}value_proj.bias", f"{ca_post}value_proj.bias"))
        append((f"{ca_pre}output_proj.weight", f"{ca_post}output_proj.weight"))
        append((f"{ca_pre}output_proj.bias", f"{ca_post}output_proj.bias"))
        append((f"{dec_pre}.norm1.weight", f"{dec_post}.self_attn_layer_norm.weight"))
        append((f"{dec_pre}.norm1.bias", f"{dec_post}.self_attn_layer_norm.bias"))
        append((f"{dec_pre}.norm2.weight", f"{dec_post}.encoder_attn_layer_norm.weight"))
        append((f"{dec_pre}.norm2.bias", f"{dec_post}.encoder_attn_layer_norm.bias"))
        append((f"{dec_pre}.linear1.weight", f"{dec_post}.fc1.weight"))
        append((f"{dec_pre}.linear1.bias", f"{dec_post}.fc1.bias"))
        append((f"{dec_pre}.linear2.weight", f"{dec_post}.fc2.weight"))
        append((f"{dec_pre}.linear2.bias", f"{dec_post}.fc2.bias"))
        append((f"{dec_pre}.norm3.weight", f"{dec_post}.final_layer_norm.weight"))
        append((f"{dec_pre}.norm3.bias", f"{dec_post}.final_layer_norm.bias"))

    for i in range(decoder_layers):
        ce_pre = f"decoder.dec_score_head.{i}."
        ce_post = f"model.decoder.class_embed.{i}."
        append((f"{ce_pre}weight", f"{ce_post}weight"))
        append((f"{ce_pre}bias", f"{ce_post}bias"))
        be_pre = f"decoder.dec_bbox_head.{i}.layers."
        be_post = f"model.decoder.bbox_embed.{i}.layers."
        append((f"{be_pre}0.weight", f"{be_post}0.weight"))
        append((f"{be_pre}0.bias", f"{be_post}0.bias"))
        append((f"{be_pre}1.weight", f"{be_post}1.weight"))
        append((f"{be_pre}1.bias", f"{be_post}1.bias"))
        append((f"{be_pre}2.weight", f"{be_post}2.weight"))
        append((f"{be_pre}2.bias", f"{be_post}2.bias"))

    # decoder input proj
    for i in range(len_decoder_in_channels):
        append((f"decoder.input_proj.{i}.conv.weight", f"model.decoder_input_proj.{i}.0.weight"))
        norm_pre = f"decoder.input_proj.{i}.norm."
        norm_post = f"model.decoder_input_proj.{i}.1."
        append((f"{norm_pre}weight", f"{norm_post}weight"))
        append((f"{norm_pre}bias", f"{norm_post}bias"))
        append((f"{norm_pre}running_mean", f"{norm_post}running_mean"))
        append((f"{norm_pre}running_var", f"{norm_post}running_var"))

    # convolutional projection + query embeddings + layernorm of decoder + class and bounding box heads
    append(("decoder.denoising_class_embed.weight", "model.denoising_class_embed.weight"))
    for i in range(2):
        append((f"decoder.query_pos_head.layers.{i}.weight", f"model.decoder.query_pos_head.layers.{i}.weight"))
        append((f"decoder.query_pos_head.layers.{i}.bias", f"model.decoder.query_pos_head.layers.{i}.bias"))
    for i in range(2):
        append((f"decoder.enc_output.{i}.weight", f"model.enc_output.{i}.weight"))
        append((f"decoder.enc_output.{i}.bias", f"model.enc_output.{i}.bias"))
    append(("decoder.enc_score_head.weight", "model.enc_score_head.weight"))
    append(("decoder.enc_score_head.bias", "model.enc_score_head.bias"))
    for i in range(3):
        append((f"decoder.enc_bbox_head.layers.{i}.weight", f"model.enc_bbox_head.layers.{i}.weight"))
        append((f"decoder.enc_bbox_head.layers.{i}.bias", f"model.enc_bbox_head.layers.{i}.bias"))

    return rename_keys


def rename_key(state_dict, old, new):
    try:
        val = state_dict.pop(old)
        state_dict[new] = val
    except Exception:
        pass


def read_in_q_k_v(state_dict, config):
    prefix = ""
    encoder_hidden_dim = config.encoder_hidden_dim

    # first: transformer encoder
    for i in range(config.encoder_layers):
        # read in weights + bias of input projection layer (in PyTorch's MultiHeadAttention, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"{prefix}encoder.encoder.{i}.layers.0.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}encoder.encoder.{i}.layers.0.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.q_proj.weight"] = in_proj_weight[
            :encoder_hidden_dim, :
        ]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.q_proj.bias"] = in_proj_bias[:encoder_hidden_dim]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.k_proj.weight"] = in_proj_weight[
            encoder_hidden_dim : 2 * encoder_hidden_dim, :
        ]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.k_proj.bias"] = in_proj_bias[
            encoder_hidden_dim : 2 * encoder_hidden_dim
        ]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.v_proj.weight"] = in_proj_weight[
            -encoder_hidden_dim:, :
        ]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.v_proj.bias"] = in_proj_bias[-encoder_hidden_dim:]
    # next: transformer decoder (which is a bit more complex because it also includes cross-attention)
    for i in range(config.decoder_layers):
        # read in weights + bias of input projection layer of self-attention
        in_proj_weight = state_dict.pop(f"{prefix}decoder.decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}decoder.decoder.layers.{i}.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)

    return im


@torch.no_grad()
def convert_rt_detr_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub, repo_id):
    """
    Copy/paste/tweak model's weights to our RTDETR structure.
    """

    # load default config
    config = get_rt_detr_config(model_name)

    # load original model from torch hub
    model_name_to_checkpoint_url = {
        "rtdetr_r18vd": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth",
        "rtdetr_r34vd": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth",
        "rtdetr_r50vd_m": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_m_6x_coco_from_paddle.pth",
        "rtdetr_r50vd": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth",
        "rtdetr_r101vd": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_6x_coco_from_paddle.pth",
        "rtdetr_r18vd_coco_o365": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth",
        "rtdetr_r50vd_coco_o365": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth",
        "rtdetr_r101vd_coco_o365": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth",
    }
    logger.info(f"Converting model {model_name}...")
    state_dict = torch.hub.load_state_dict_from_url(model_name_to_checkpoint_url[model_name], map_location="cpu")[
        "ema"
    ]["module"]

    # rename keys
    for src, dest in create_rename_keys(config):
        rename_key(state_dict, src, dest)
    # query, key and value matrices need special treatment
    read_in_q_k_v(state_dict, config)
    # important: we need to prepend a prefix to each of the base model keys as the head models use different attributes for them
    for key in state_dict.copy():
        if key.endswith("num_batches_tracked"):
            del state_dict[key]
        # for two_stage
        if "bbox_embed" in key or ("class_embed" in key and "denoising_" not in key):
            state_dict[key.split("model.decoder.")[-1]] = state_dict[key]

    # finally, create HuggingFace model and load state dict
    model = RTDetrForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()

    # load image processor
    image_processor = RTDetrImageProcessor()

    # prepare image
    img = prepare_img()

    # preprocess image
    transformations = transforms.Compose(
        [
            transforms.Resize([640, 640], interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )
    original_pixel_values = transformations(img).unsqueeze(0)  # insert batch dimension

    encoding = image_processor(images=img, return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    assert torch.allclose(original_pixel_values, pixel_values)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pixel_values = pixel_values.to(device)

    # Pass image by the model
    outputs = model(pixel_values)

    if model_name == "rtdetr_r18vd":
        expected_slice_logits = torch.tensor(
            [
                [-4.3364253, -6.465683, -3.6130402],
                [-4.083815, -6.4039373, -6.97881],
                [-4.192215, -7.3410473, -6.9027247],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.16868353, 0.19833282, 0.21182671],
                [0.25559652, 0.55121744, 0.47988364],
                [0.7698693, 0.4124569, 0.46036878],
            ]
        )
    elif model_name == "rtdetr_r34vd":
        expected_slice_logits = torch.tensor(
            [
                [-4.3727384, -4.7921476, -5.7299604],
                [-4.840536, -8.455345, -4.1745796],
                [-4.1277084, -5.2154565, -5.7852697],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.258278, 0.5497808, 0.4732004],
                [0.16889669, 0.19890057, 0.21138911],
                [0.76632994, 0.4147879, 0.46851268],
            ]
        )
    elif model_name == "rtdetr_r50vd_m":
        expected_slice_logits = torch.tensor(
            [
                [-4.319764, -6.1349025, -6.094794],
                [-5.1056995, -7.744766, -4.803956],
                [-4.7685347, -7.9278393, -4.5751696],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.2582739, 0.55071366, 0.47660282],
                [0.16811174, 0.19954777, 0.21292639],
                [0.54986024, 0.2752091, 0.0561416],
            ]
        )
    elif model_name == "rtdetr_r50vd":
        expected_slice_logits = torch.tensor(
            [
                [-4.6476398, -5.001154, -4.9785104],
                [-4.1593494, -4.7038546, -5.946485],
                [-4.4374595, -4.658361, -6.2352347],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.16880608, 0.19992264, 0.21225442],
                [0.76837635, 0.4122631, 0.46368608],
                [0.2595386, 0.5483334, 0.4777486],
            ]
        )
    elif model_name == "rtdetr_r101vd":
        expected_slice_logits = torch.tensor(
            [
                [-4.6162, -4.9189, -4.6656],
                [-4.4701, -4.4997, -4.9659],
                [-5.6641, -7.9000, -5.0725],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.7707, 0.4124, 0.4585],
                [0.2589, 0.5492, 0.4735],
                [0.1688, 0.1993, 0.2108],
            ]
        )
    elif model_name == "rtdetr_r18vd_coco_o365":
        expected_slice_logits = torch.tensor(
            [
                [-4.8726, -5.9066, -5.2450],
                [-4.8157, -6.8764, -5.1656],
                [-4.7492, -5.7006, -5.1333],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.2552, 0.5501, 0.4773],
                [0.1685, 0.1986, 0.2104],
                [0.7692, 0.4141, 0.4620],
            ]
        )
    elif model_name == "rtdetr_r50vd_coco_o365":
        expected_slice_logits = torch.tensor(
            [
                [-4.6491, -3.9252, -5.3163],
                [-4.1386, -5.0348, -3.9016],
                [-4.4778, -4.5423, -5.7356],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.2583, 0.5492, 0.4747],
                [0.5501, 0.2754, 0.0574],
                [0.7693, 0.4137, 0.4613],
            ]
        )
    elif model_name == "rtdetr_r101vd_coco_o365":
        expected_slice_logits = torch.tensor(
            [
                [-4.5152, -5.6811, -5.7311],
                [-4.5358, -7.2422, -5.0941],
                [-4.6919, -5.5834, -6.0145],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.7703, 0.4140, 0.4583],
                [0.1686, 0.1991, 0.2107],
                [0.2570, 0.5496, 0.4750],
            ]
        )
    else:
        raise ValueError(f"Unknown rt_detr_name: {model_name}")

    assert torch.allclose(outputs.logits[0, :3, :3], expected_slice_logits.to(outputs.logits.device), atol=1e-4)
    assert torch.allclose(outputs.pred_boxes[0, :3, :3], expected_slice_boxes.to(outputs.pred_boxes.device), atol=1e-3)

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        print(f"Saving image processor to {pytorch_dump_folder_path}")
        image_processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        # Upload model, image processor and config to the hub
        logger.info("Uploading PyTorch model and image processor to the hub...")
        config.push_to_hub(
            repo_id=repo_id, commit_message="Add config from convert_rt_detr_original_pytorch_checkpoint_to_pytorch.py"
        )
        model.push_to_hub(
            repo_id=repo_id, commit_message="Add model from convert_rt_detr_original_pytorch_checkpoint_to_pytorch.py"
        )
        image_processor.push_to_hub(
            repo_id=repo_id,
            commit_message="Add image processor from convert_rt_detr_original_pytorch_checkpoint_to_pytorch.py",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="rtdetr_r50vd",
        type=str,
        help="model_name of the checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model to the hub or not.")
    parser.add_argument(
        "--repo_id",
        type=str,
        help="repo_id where the model will be pushed to.",
    )
    args = parser.parse_args()
    convert_rt_detr_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub, args.repo_id)
