"""Optional visible-token prediction loss for frame-token MotionJEPA."""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F

from mask.utils import index_mask_validity, repeat_mask_blocks


def context_loss_config(config: dict) -> dict:
    """Resolve and validate configuration without modifying the caller."""
    losses = config.get("loss", {})
    if not isinstance(losses, Mapping):
        raise ValueError("loss config must be a mapping")
    options = losses.get("context", {})
    if not isinstance(options, Mapping):
        raise ValueError("loss.context config must be a mapping")
    resolved = {}
    for key, default in (("weight", 0.0), ("warmup_epochs", 40.0)):
        value = options.get(key, default)
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"loss.context.{key} must be finite and nonnegative") from error
        if isinstance(value, bool) or not math.isfinite(number) or number < 0:
            raise ValueError(f"loss.context.{key} must be finite and nonnegative")
        resolved[key] = number
    weighting = options.get("distance_weighting", True)
    if not isinstance(weighting, bool):
        raise ValueError("loss.context.distance_weighting must be a boolean")
    resolved["distance_weighting"] = weighting
    return resolved


def validate_context_loss_layout(options: dict, *layouts) -> None:
    if options["weight"] > 0 and any(
        layout.kind != "1d" or layout.patchified for layout in layouts
    ):
        raise ValueError("Positive context loss is supported only for non-patchified 1D models")


def context_loss_weight(options: dict, global_step: int, iterations_per_epoch: int) -> float:
    """Coefficient for the next update; restored global_step counts completed updates."""
    if options["weight"] == 0:
        return 0.0
    duration = options["warmup_epochs"] * iterations_per_epoch
    if not math.isfinite(duration) or iterations_per_epoch <= 0 or global_step < 0:
        raise ValueError("Invalid context-loss warmup duration or training step")
    steps = math.ceil(duration)
    return options["weight"] * min(1.0, global_step / steps) if steps else options["weight"]


def context_distance_weights(
    masks_enc: list[torch.Tensor], masks_pred: list[torch.Tensor]
) -> torch.Tensor:
    """Nearest target distance in original frame indices, ordered [pred, enc, batch]."""
    batch_size = masks_enc[0].shape[0]
    context = torch.cat(masks_enc, dim=0).long().repeat(len(masks_pred), 1)
    target = repeat_mask_blocks(torch.cat(masks_pred, dim=0).long(), batch_size, len(masks_enc))
    distances = (context.unsqueeze(-1) - target.unsqueeze(1)).abs().float()
    distances = distances.masked_fill(target.unsqueeze(1) < 0, float("inf"))
    nearest = distances.amin(dim=-1)
    return nearest.clamp_min(1).rsqrt().masked_fill(context < 0, 0)


def context_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    masks_enc: list[torch.Tensor],
    masks_pred: list[torch.Tensor],
    *,
    distance_weighting: bool,
) -> torch.Tensor:
    """Smooth-L1 averaged equally over samples and every prediction/context view."""
    context_active = index_mask_validity(masks_enc)
    target_active = index_mask_validity(masks_pred)
    batch_size = masks_enc[0].shape[0]
    if masks_pred[0].shape[0] != batch_size:
        raise ValueError("Context and target masks must use the same batch size")
    if not context_active.any(dim=1).all() or not target_active.any(dim=1).all():
        raise ValueError("Each context and target mask row must contain a valid token")
    active = context_active.repeat(len(masks_pred), 1).to(prediction.device)
    if prediction.ndim != 3 or prediction.shape != target.shape or active.shape != prediction.shape[:2]:
        raise ValueError("Context prediction, target, and mask shapes do not match")
    # Remove padded values before Smooth-L1, including possible non-finite sentinels.
    prediction = prediction.masked_fill(~active.unsqueeze(-1), 0)
    target = target.detach().masked_fill(~active.unsqueeze(-1), 0)
    errors = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
    if distance_weighting:
        errors = errors * context_distance_weights(masks_enc, masks_pred).to(errors.device)
    errors = errors.masked_fill(~active, 0)
    # Deliberately divide by token count, not by the sum of distance weights.
    return (errors.sum(dim=1) / active.sum(dim=1)).mean()
