# Copyright 2025 The HuggingFace Team. All rights reserved.
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


import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import is_vision_available
from .loss_for_object_detection import box_iou
from .loss_rt_detr import RTDetrHungarianMatcher, RTDetrLoss


if is_vision_available():
    from transformers.image_transforms import center_to_corners_format


def _set_aux_loss(outputs_class, outputs_coord):
    return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class, outputs_coord)]


def _set_aux_loss2(
    outputs_class, outputs_coord, outputs_corners, outputs_ref, teacher_corners=None, teacher_logits=None
):
    return [
        {
            "logits": a,
            "pred_boxes": b,
            "pred_corners": c,
            "ref_points": d,
            "teacher_corners": teacher_corners,
            "teacher_logits": teacher_logits,
        }
        for a, b, c, d in zip(outputs_class, outputs_coord, outputs_corners, outputs_ref)
    ]


def weighting_function(max_num_bins: int, up: torch.Tensor, reg_scale: int) -> torch.Tensor:
    """
    Generates the non-uniform Weighting Function W(n) for bounding box regression.

    Args:
        max_num_bins (int): Max number of the discrete bins.
        up (Tensor): Controls upper bounds of the sequence,
                     where maximum offset is ±up * H / W.
        reg_scale (float): Controls the curvature of the Weighting Function.
                           Larger values result in flatter weights near the central axis W(max_num_bins/2)=0
                           and steeper weights at both ends.
    Returns:
        Tensor: Sequence of Weighting Function.
    """
    upper_bound1 = abs(up[0]) * abs(reg_scale)
    upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
    # Precompute exponents and powers for vectorization
    step_power = 2 / (max_num_bins - 2)
    step = (upper_bound1 + 1) ** step_power

    half = max_num_bins // 2
    device = up.device
    dtype = up.dtype

    # Vectorize left_values and right_values (these are non-tensor objects in orig, but can be made tensors)
    if half > 1:
        left_inds = torch.arange(half - 1, 0, -1, dtype=dtype, device=device)
        left_values = -(step**left_inds) + 1
    else:
        left_values = torch.empty(0, dtype=dtype, device=device)
    if half > 0:
        right_inds = torch.arange(1, half, dtype=dtype, device=device)
        right_values = (step**right_inds) - 1
    else:
        right_values = torch.empty(0, dtype=dtype, device=device)
    zero_value = torch.zeros(1, dtype=dtype, device=device)
    values_parts = [
        torch.as_tensor([-upper_bound2], dtype=dtype, device=device),
        left_values,
        zero_value,
        right_values,
        torch.as_tensor([upper_bound2], dtype=dtype, device=device),
    ]
    values = torch.cat(values_parts, dim=0)
    return values


def translate_gt(gt: torch.Tensor, max_num_bins: int, reg_scale: int, up: torch.Tensor):
    """
    Decodes bounding box ground truth (GT) values into distribution-based GT representations.

    This function maps continuous GT values into discrete distribution bins, which can be used
    for regression tasks in object detection models. It calculates the indices of the closest
    bins to each GT value and assigns interpolation weights to these bins based on their proximity
    to the GT value.

    Args:
        gt (Tensor): Ground truth bounding box values, shape (N, ).
        max_num_bins (int): Maximum number of discrete bins for the distribution.
        reg_scale (float): Controls the curvature of the Weighting Function.
        up (Tensor): Controls the upper bounds of the Weighting Function.

    Returns:
        tuple[Tensor, Tensor, Tensor]:
            - indices (Tensor): Index of the left bin closest to each GT value, shape (N, ).
            - weight_right (Tensor): Weight assigned to the right bin, shape (N, ).
            - weight_left (Tensor): Weight assigned to the left bin, shape (N, ).
    """
    gt = gt.reshape(-1)
    function_values = weighting_function(max_num_bins, up, reg_scale)
    # Use torch.searchsorted for left indices (vectorized, monotonic increasing)
    # function_values assumed sorted ascending
    closest_left_indices = torch.searchsorted(function_values, gt, right=True) - 1

    # Calculate the weights for the interpolation
    indices = closest_left_indices.float()

    weight_right = torch.zeros_like(indices)
    weight_left = torch.zeros_like(indices)

    valid_idx_mask = (indices >= 0) & (indices < max_num_bins)
    valid_indices = indices[valid_idx_mask].long()

    # Only perform gather if any valid indices
    if valid_indices.numel() > 0:
        left_values = function_values[valid_indices]
        right_values = function_values[valid_indices + 1]

        left_diffs = torch.abs(gt[valid_idx_mask] - left_values)
        right_diffs = torch.abs(right_values - gt[valid_idx_mask])

        denom = left_diffs + right_diffs
        # denom > 0 guaranteed (since left!=right unless gt is precisely on value, but safe /0 handled by 0/0=nan)
        weight_right_valid = left_diffs / denom
        weight_right[valid_idx_mask] = weight_right_valid
        weight_left[valid_idx_mask] = 1.0 - weight_right_valid

    # Invalid weights (out of range)

    # Invalid weights (out of range)
    invalid_idx_mask_neg = indices < 0
    if invalid_idx_mask_neg.any():
        weight_right[invalid_idx_mask_neg] = 0.0
        weight_left[invalid_idx_mask_neg] = 1.0
        indices[invalid_idx_mask_neg] = 0.0

    invalid_idx_mask_pos = indices >= max_num_bins
    if invalid_idx_mask_pos.any():
        weight_right[invalid_idx_mask_pos] = 1.0
        weight_left[invalid_idx_mask_pos] = 0.0
        indices[invalid_idx_mask_pos] = max_num_bins - 0.1

    return indices, weight_right, weight_left


def bbox2distance(points, bbox, max_num_bins, reg_scale, up, eps=0.1):
    """
    Converts bounding box coordinates to distances from a reference point.

    Args:
        points (Tensor): (n, 4) [x, y, w, h], where (x, y) is the center.
        bbox (Tensor): (n, 4) bounding boxes in "xyxy" format.
        max_num_bins (float): Maximum bin value.
        reg_scale (float): Controlling curvarture of W(n).
        up (Tensor): Controlling upper bounds of W(n).
        eps (float): Small value to ensure target < max_num_bins.

    Returns:
        Tensor: Decoded distances.
    """

    reg_scale = abs(reg_scale)
    left = (points[:, 0] - bbox[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    top = (points[:, 1] - bbox[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    right = (bbox[:, 2] - points[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    bottom = (bbox[:, 3] - points[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    four_lens = torch.stack([left, top, right, bottom], -1)
    four_lens, weight_right, weight_left = translate_gt(four_lens, max_num_bins, reg_scale, up)
    if max_num_bins is not None:
        four_lens = four_lens.clamp(min=0, max=max_num_bins - eps)
    return four_lens.reshape(-1).detach(), weight_right.detach(), weight_left.detach()


class DFineLoss(RTDetrLoss):
    """
    This class computes the losses for D-FINE. The process happens in two steps: 1) we compute hungarian assignment
    between ground truth boxes and the outputs of the model 2) we supervise each pair of matched ground-truth /
    prediction (supervise class and box).

    Args:
        matcher (`DetrHungarianMatcher`):
            Module able to compute a matching between targets and proposals.
        weight_dict (`Dict`):
            Dictionary relating each loss with its weights. These losses are configured in DFineConf as
            `weight_loss_vfl`, `weight_loss_bbox`, `weight_loss_giou`, `weight_loss_fgl`, `weight_loss_ddf`
        losses (`list[str]`):
            List of all the losses to be applied. See `get_loss` for a list of all available losses.
        alpha (`float`):
            Parameter alpha used to compute the focal loss.
        gamma (`float`):
            Parameter gamma used to compute the focal loss.
        eos_coef (`float`):
            Relative classification weight applied to the no-object category.
        num_classes (`int`):
            Number of object categories, omitting the special no-object category.
    """

    def __init__(self, config):
        super().__init__(config)

        self.matcher = RTDetrHungarianMatcher(config)
        self.max_num_bins = config.max_num_bins
        self.weight_dict = {
            "loss_vfl": config.weight_loss_vfl,
            "loss_bbox": config.weight_loss_bbox,
            "loss_giou": config.weight_loss_giou,
            "loss_fgl": config.weight_loss_fgl,
            "loss_ddf": config.weight_loss_ddf,
        }
        self.losses = ["vfl", "boxes", "local"]
        self.reg_scale = config.reg_scale
        self.up = nn.Parameter(torch.tensor([config.up]), requires_grad=False)

    def unimodal_distribution_focal_loss(
        self, pred, label, weight_right, weight_left, weight=None, reduction="sum", avg_factor=None
    ):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction="none") * weight_left.reshape(-1) + F.cross_entropy(
            pred, dis_right, reduction="none"
        ) * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """Compute Fine-Grained Localization (FGL) Loss
        and Decoupled Distillation Focal (DDF) Loss."""

        losses = {}
        if "pred_corners" in outputs:
            idx = self._get_source_permutation_idx(indices)
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

            pred_corners = outputs["pred_corners"][idx].reshape(-1, (self.max_num_bins + 1))
            ref_points = outputs["ref_points"][idx].detach()
            with torch.no_grad():
                self.fgl_targets = bbox2distance(
                    ref_points,
                    center_to_corners_format(target_boxes),
                    self.max_num_bins,
                    self.reg_scale,
                    self.up,
                )

            target_corners, weight_right, weight_left = self.fgl_targets

            ious = torch.diag(
                box_iou(center_to_corners_format(outputs["pred_boxes"][idx]), center_to_corners_format(target_boxes))[
                    0
                ]
            )
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            losses["loss_fgl"] = self.unimodal_distribution_focal_loss(
                pred_corners,
                target_corners,
                weight_right,
                weight_left,
                weight_targets,
                avg_factor=num_boxes,
            )

            pred_corners = outputs["pred_corners"].reshape(-1, (self.max_num_bins + 1))
            target_corners = outputs["teacher_corners"].reshape(-1, (self.max_num_bins + 1))
            if torch.equal(pred_corners, target_corners):
                losses["loss_ddf"] = pred_corners.sum() * 0
            else:
                weight_targets_local = outputs["teacher_logits"].sigmoid().max(dim=-1)[0]
                mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                mask[idx] = True
                mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(weight_targets_local.dtype)
                weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

                loss_match_local = (
                    weight_targets_local
                    * (T**2)
                    * (
                        nn.KLDivLoss(reduction="none")(
                            F.log_softmax(pred_corners / T, dim=1),
                            F.softmax(target_corners.detach() / T, dim=1),
                        )
                    ).sum(-1)
                )

                batch_scale = 1 / outputs["pred_boxes"].shape[0]  # it should be refined
                self.num_pos, self.num_neg = (
                    (mask.sum() * batch_scale) ** 0.5,
                    ((~mask).sum() * batch_scale) ** 0.5,
                )
                loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                losses["loss_ddf"] = (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg) / (
                    self.num_pos + self.num_neg
                )

        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            "cardinality": self.loss_cardinality,
            "local": self.loss_local,
            "boxes": self.loss_boxes,
            "focal": self.loss_labels_focal,
            "vfl": self.loss_labels_vfl,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_boxes)


def DFineForObjectDetectionLoss(
    logits,
    labels,
    device,
    pred_boxes,
    config,
    outputs_class=None,
    outputs_coord=None,
    enc_topk_logits=None,
    enc_topk_bboxes=None,
    denoising_meta_values=None,
    predicted_corners=None,
    initial_reference_points=None,
    **kwargs,
):
    criterion = DFineLoss(config)
    criterion.to(device)
    # Second: compute the losses, based on outputs and labels
    outputs_loss = {}
    outputs_loss["logits"] = logits
    outputs_loss["pred_boxes"] = pred_boxes.clamp(min=0, max=1)
    auxiliary_outputs = None
    if config.auxiliary_loss:
        if denoising_meta_values is not None:
            dn_out_coord, outputs_coord = torch.split(
                outputs_coord.clamp(min=0, max=1), denoising_meta_values["dn_num_split"], dim=2
            )
            dn_out_class, outputs_class = torch.split(outputs_class, denoising_meta_values["dn_num_split"], dim=2)
            dn_out_corners, out_corners = torch.split(predicted_corners, denoising_meta_values["dn_num_split"], dim=2)
            dn_out_refs, out_refs = torch.split(initial_reference_points, denoising_meta_values["dn_num_split"], dim=2)

            auxiliary_outputs = _set_aux_loss2(
                outputs_class[:, :-1].transpose(0, 1),
                outputs_coord[:, :-1].transpose(0, 1),
                out_corners[:, :-1].transpose(0, 1),
                out_refs[:, :-1].transpose(0, 1),
                out_corners[:, -1],
                outputs_class[:, -1],
            )

            outputs_loss["auxiliary_outputs"] = auxiliary_outputs
            outputs_loss["auxiliary_outputs"].extend(
                _set_aux_loss([enc_topk_logits], [enc_topk_bboxes.clamp(min=0, max=1)])
            )

            dn_auxiliary_outputs = _set_aux_loss2(
                dn_out_class.transpose(0, 1),
                dn_out_coord.transpose(0, 1),
                dn_out_corners.transpose(0, 1),
                dn_out_refs.transpose(0, 1),
                dn_out_corners[:, -1],
                dn_out_class[:, -1],
            )
            outputs_loss["dn_auxiliary_outputs"] = dn_auxiliary_outputs
            outputs_loss["denoising_meta_values"] = denoising_meta_values

    loss_dict = criterion(outputs_loss, labels)

    loss = sum(loss_dict.values())
    return loss, loss_dict, auxiliary_outputs
