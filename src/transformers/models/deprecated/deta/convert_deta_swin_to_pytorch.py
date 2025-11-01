# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""Convert DETA checkpoints from the original repository.

URL: https://github.com/jozhang97/DETA/tree/master"""

import argparse
import json
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import DetaConfig, DetaForObjectDetection, DetaImageProcessor, SwinConfig
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_deta_config(model_name):
    backbone_config = SwinConfig(
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        window_size=12,
        out_features=["stage2", "stage3", "stage4"],
    )

    config = DetaConfig(
        backbone_config=backbone_config,
        num_queries=900,
        encoder_ffn_dim=2048,
        decoder_ffn_dim=2048,
        num_feature_levels=5,
        assign_first_stage=True,
        with_box_refine=True,
        two_stage=True,
    )

    # set labels
    repo_id = "huggingface/label-files"
    if "o365" in model_name:
        num_labels = 366
        filename = "object365-id2label.json"
    else:
        num_labels = 91
        filename = "coco-detection-id2label.json"

    config.num_labels = num_labels
    id2label = json.loads(Path(hf_hub_download(repo_id, filename, repo_type="dataset")).read_text())
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    return config


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config):
    # Use local names to avoid repeated attribute lookups
    backbone_depths = config.backbone_config.depths
    encoder_layers = config.encoder_layers
    decoder_layers = config.decoder_layers

    rename_keys = [
        # stem
        (
            "backbone.0.body.patch_embed.proj.weight",
            "model.backbone.model.embeddings.patch_embeddings.projection.weight",
        ),
        ("backbone.0.body.patch_embed.proj.bias", "model.backbone.model.embeddings.patch_embeddings.projection.bias"),
        ("backbone.0.body.patch_embed.norm.weight", "model.backbone.model.embeddings.norm.weight"),
        ("backbone.0.body.patch_embed.norm.bias", "model.backbone.model.embeddings.norm.bias"),
    ]
    # stages
    # Preallocate capacity if available (optimize list growth for large configs, optional in CPython)
    append = rename_keys.append  # Minor speedup for tight loops

    for i, num_blocks in enumerate(backbone_depths):
        # Unroll block-level key creation and append using direct access
        for j in range(num_blocks):
            prefix = f"backbone.0.body.layers.{i}.blocks.{j}"
            enc_prefix = f"model.backbone.model.encoder.layers.{i}.blocks.{j}"
            append((f"{prefix}.norm1.weight", f"{enc_prefix}.layernorm_before.weight"))
            append((f"{prefix}.norm1.bias", f"{enc_prefix}.layernorm_before.bias"))
            append(
                (
                    f"{prefix}.attn.relative_position_bias_table",
                    f"{enc_prefix}.attention.self.relative_position_bias_table",
                )
            )
            append((f"{prefix}.attn.relative_position_index", f"{enc_prefix}.attention.self.relative_position_index"))
            append((f"{prefix}.attn.proj.weight", f"{enc_prefix}.attention.output.dense.weight"))
            append((f"{prefix}.attn.proj.bias", f"{enc_prefix}.attention.output.dense.bias"))
            append((f"{prefix}.norm2.weight", f"{enc_prefix}.layernorm_after.weight"))
            append((f"{prefix}.norm2.bias", f"{enc_prefix}.layernorm_after.bias"))
            append((f"{prefix}.mlp.fc1.weight", f"{enc_prefix}.intermediate.dense.weight"))
            append((f"{prefix}.mlp.fc1.bias", f"{enc_prefix}.intermediate.dense.bias"))
            append((f"{prefix}.mlp.fc2.weight", f"{enc_prefix}.output.dense.weight"))
            append((f"{prefix}.mlp.fc2.bias", f"{enc_prefix}.output.dense.bias"))

        if i < 3:
            layer_prefix = f"backbone.0.body.layers.{i}.downsample"
            enc_layer_prefix = f"model.backbone.model.encoder.layers.{i}.downsample"
            append((f"{layer_prefix}.reduction.weight", f"{enc_layer_prefix}.reduction.weight"))
            append((f"{layer_prefix}.norm.weight", f"{enc_layer_prefix}.norm.weight"))
            append((f"{layer_prefix}.norm.bias", f"{enc_layer_prefix}.norm.bias"))

    rename_keys.extend(
        [
            ("backbone.0.body.norm1.weight", "model.backbone.model.hidden_states_norms.stage2.weight"),
            ("backbone.0.body.norm1.bias", "model.backbone.model.hidden_states_norms.stage2.bias"),
            ("backbone.0.body.norm2.weight", "model.backbone.model.hidden_states_norms.stage3.weight"),
            ("backbone.0.body.norm2.bias", "model.backbone.model.hidden_states_norms.stage3.bias"),
            ("backbone.0.body.norm3.weight", "model.backbone.model.hidden_states_norms.stage4.weight"),
            ("backbone.0.body.norm3.bias", "model.backbone.model.hidden_states_norms.stage4.bias"),
        ]
    )

    # transformer encoder
    for i in range(encoder_layers):
        encp = f"transformer.encoder.layers.{i}"
        tgtp = f"model.encoder.layers.{i}"
        append((f"{encp}.self_attn.sampling_offsets.weight", f"{tgtp}.self_attn.sampling_offsets.weight"))
        append((f"{encp}.self_attn.sampling_offsets.bias", f"{tgtp}.self_attn.sampling_offsets.bias"))
        append((f"{encp}.self_attn.attention_weights.weight", f"{tgtp}.self_attn.attention_weights.weight"))
        append((f"{encp}.self_attn.attention_weights.bias", f"{tgtp}.self_attn.attention_weights.bias"))
        append((f"{encp}.self_attn.value_proj.weight", f"{tgtp}.self_attn.value_proj.weight"))
        append((f"{encp}.self_attn.value_proj.bias", f"{tgtp}.self_attn.value_proj.bias"))
        append((f"{encp}.self_attn.output_proj.weight", f"{tgtp}.self_attn.output_proj.weight"))
        append((f"{encp}.self_attn.output_proj.bias", f"{tgtp}.self_attn.output_proj.bias"))
        append((f"{encp}.norm1.weight", f"{tgtp}.self_attn_layer_norm.weight"))
        append((f"{encp}.norm1.bias", f"{tgtp}.self_attn_layer_norm.bias"))
        append((f"{encp}.linear1.weight", f"{tgtp}.fc1.weight"))
        append((f"{encp}.linear1.bias", f"{tgtp}.fc1.bias"))
        append((f"{encp}.linear2.weight", f"{tgtp}.fc2.weight"))
        append((f"{encp}.linear2.bias", f"{tgtp}.fc2.bias"))
        append((f"{encp}.norm2.weight", f"{tgtp}.final_layer_norm.weight"))
        append((f"{encp}.norm2.bias", f"{tgtp}.final_layer_norm.bias"))

    # transformer decoder
    for i in range(decoder_layers):
        decp = f"transformer.decoder.layers.{i}"
        tgtp = f"model.decoder.layers.{i}"
        append((f"{decp}.cross_attn.sampling_offsets.weight", f"{tgtp}.encoder_attn.sampling_offsets.weight"))
        append((f"{decp}.cross_attn.sampling_offsets.bias", f"{tgtp}.encoder_attn.sampling_offsets.bias"))
        append((f"{decp}.cross_attn.attention_weights.weight", f"{tgtp}.encoder_attn.attention_weights.weight"))
        append((f"{decp}.cross_attn.attention_weights.bias", f"{tgtp}.encoder_attn.attention_weights.bias"))
        append((f"{decp}.cross_attn.value_proj.weight", f"{tgtp}.encoder_attn.value_proj.weight"))
        append((f"{decp}.cross_attn.value_proj.bias", f"{tgtp}.encoder_attn.value_proj.bias"))
        append((f"{decp}.cross_attn.output_proj.weight", f"{tgtp}.encoder_attn.output_proj.weight"))
        append((f"{decp}.cross_attn.output_proj.bias", f"{tgtp}.encoder_attn.output_proj.bias"))
        append((f"{decp}.norm1.weight", f"{tgtp}.encoder_attn_layer_norm.weight"))
        append((f"{decp}.norm1.bias", f"{tgtp}.encoder_attn_layer_norm.bias"))
        append((f"{decp}.self_attn.out_proj.weight", f"{tgtp}.self_attn.out_proj.weight"))
        append((f"{decp}.self_attn.out_proj.bias", f"{tgtp}.self_attn.out_proj.bias"))
        append((f"{decp}.norm2.weight", f"{tgtp}.self_attn_layer_norm.weight"))
        append((f"{decp}.norm2.bias", f"{tgtp}.self_attn_layer_norm.bias"))
        append((f"{decp}.linear1.weight", f"{tgtp}.fc1.weight"))
        append((f"{decp}.linear1.bias", f"{tgtp}.fc1.bias"))
        append((f"{decp}.linear2.weight", f"{tgtp}.fc2.weight"))
        append((f"{decp}.linear2.bias", f"{tgtp}.fc2.bias"))
        append((f"{decp}.norm3.weight", f"{tgtp}.final_layer_norm.weight"))
        append((f"{decp}.norm3.bias", f"{tgtp}.final_layer_norm.bias"))

    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_swin_q_k_v(state_dict, backbone_config):
    num_features = [int(backbone_config.embed_dim * 2**i) for i in range(len(backbone_config.depths))]
    for i in range(len(backbone_config.depths)):
        dim = num_features[i]
        for j in range(backbone_config.depths[i]):
            # fmt: off
            # read in weights + bias of input projection layer (in original implementation, this is a single matrix + bias)
            in_proj_weight = state_dict.pop(f"backbone.0.body.layers.{i}.blocks.{j}.attn.qkv.weight")
            in_proj_bias = state_dict.pop(f"backbone.0.body.layers.{i}.blocks.{j}.attn.qkv.bias")
            # next, add query, keys and values (in that order) to the state dict
            state_dict[f"model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.self.query.weight"] = in_proj_weight[:dim, :]
            state_dict[f"model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.self.query.bias"] = in_proj_bias[: dim]
            state_dict[f"model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.self.key.weight"] = in_proj_weight[
                dim : dim * 2, :
            ]
            state_dict[f"model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.self.key.bias"] = in_proj_bias[
                dim : dim * 2
            ]
            state_dict[f"model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.self.value.weight"] = in_proj_weight[
                -dim :, :
            ]
            state_dict[f"model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.self.value.bias"] = in_proj_bias[-dim :]
            # fmt: on


def read_in_decoder_q_k_v(state_dict, config):
    # transformer decoder self-attention layers
    hidden_size = config.d_model
    for i in range(config.decoder_layers):
        # read in weights + bias of input projection layer of self-attention
        in_proj_weight = state_dict.pop(f"transformer.decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"transformer.decoder.layers.{i}.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:hidden_size, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:hidden_size]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[
            hidden_size : hidden_size * 2, :
        ]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[hidden_size : hidden_size * 2]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-hidden_size:, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-hidden_size:]


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)

    return im


@torch.no_grad()
def convert_deta_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    """
    Copy/paste/tweak model's weights to our DETA structure.
    """

    # load config
    config = get_deta_config(model_name)

    # load original state dict
    if model_name == "deta-swin-large":
        checkpoint_path = hf_hub_download(repo_id="nielsr/deta-checkpoints", filename="adet_swin_ft.pth")
    elif model_name == "deta-swin-large-o365":
        checkpoint_path = hf_hub_download(repo_id="jozhang97/deta-swin-l-o365", filename="deta_swin_pt_o365.pth")
    else:
        raise ValueError(f"Model name {model_name} not supported")

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model"]

    # original state dict
    for name, param in state_dict.items():
        print(name, param.shape)

    # rename keys
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_swin_q_k_v(state_dict, config.backbone_config)
    read_in_decoder_q_k_v(state_dict, config)

    # fix some prefixes
    for key in state_dict.copy():
        if "transformer.decoder.class_embed" in key or "transformer.decoder.bbox_embed" in key:
            val = state_dict.pop(key)
            state_dict[key.replace("transformer.decoder", "model.decoder")] = val
        if "input_proj" in key:
            val = state_dict.pop(key)
            state_dict["model." + key] = val
        if "level_embed" in key or "pos_trans" in key or "pix_trans" in key or "enc_output" in key:
            val = state_dict.pop(key)
            state_dict[key.replace("transformer", "model")] = val

    # finally, create HuggingFace model and load state dict
    model = DetaForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # load image processor
    processor = DetaImageProcessor(format="coco_detection")

    # verify our conversion on image
    img = prepare_img()
    encoding = processor(images=img, return_tensors="pt")
    pixel_values = encoding["pixel_values"]
    outputs = model(pixel_values.to(device))

    # verify logits
    print("Logits:", outputs.logits[0, :3, :3])
    print("Boxes:", outputs.pred_boxes[0, :3, :3])
    if model_name == "deta-swin-large":
        expected_logits = torch.tensor(
            [[-7.6308, -2.8485, -5.3737], [-7.2037, -4.5505, -4.8027], [-7.2943, -4.2611, -4.6617]]
        )
        expected_boxes = torch.tensor([[0.4987, 0.4969, 0.9999], [0.2549, 0.5498, 0.4805], [0.5498, 0.2757, 0.0569]])
    elif model_name == "deta-swin-large-o365":
        expected_logits = torch.tensor(
            [[-8.0122, -3.5720, -4.9717], [-8.1547, -3.6886, -4.6389], [-7.6610, -3.6194, -5.0134]]
        )
        expected_boxes = torch.tensor([[0.2523, 0.5549, 0.4881], [0.7715, 0.4149, 0.4601], [0.5503, 0.2753, 0.0575]])
    assert torch.allclose(outputs.logits[0, :3, :3], expected_logits.to(device), atol=1e-4)
    assert torch.allclose(outputs.pred_boxes[0, :3, :3], expected_boxes.to(device), atol=1e-4)
    print("Everything ok!")

    if pytorch_dump_folder_path:
        # Save model and processor
        logger.info(f"Saving PyTorch model and processor to {pytorch_dump_folder_path}...")
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    # Push to hub
    if push_to_hub:
        print("Pushing model and processor to hub...")
        model.push_to_hub(f"jozhang97/{model_name}")
        processor.push_to_hub(f"jozhang97/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="deta-swin-large",
        choices=["deta-swin-large", "deta-swin-large-o365"],
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the folder to output PyTorch model.",
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    args = parser.parse_args()
    convert_deta_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
