"""S-JEPA-style distribution matching on masked frame representations."""
from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def centering_sharpening_config(config: dict) -> dict | None:
    settings = config.get("loss", {}).get("centering_sharpening", {})
    if not isinstance(settings, dict):
        raise ValueError("loss.centering_sharpening must be a mapping")
    enabled = settings.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("loss.centering_sharpening.enabled must be a boolean")
    if not enabled:
        return None
    result = {}
    for name, default in (("center_momentum", 0.9), ("teacher_temperature", 0.06),
                          ("student_temperature", 0.1)):
        value = settings.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"centering_sharpening.{name} must be finite")
        if (name == "center_momentum" and not 0 <= value < 1) or (name != "center_momentum" and value <= 0):
            raise ValueError(f"Invalid centering_sharpening.{name}: {value}")
        result[name] = float(value)
    return result


class CenteredSharpenedLoss(nn.Module):
    """Center only the teacher; compare channel distributions with cross-entropy.

    Inputs are existing predictor/teacher features, with no additional head or
    target LayerNorm. Loss weights each mask/sample row equally. The channel
    center averages valid masked target tokens across all ranks; overlapping
    blocks contribute separately, as they do to the prediction objective.
    """

    def __init__(self, feature_dim: int, *, center_momentum: float = 0.9,
                 teacher_temperature: float = 0.06, student_temperature: float = 0.1):
        super().__init__()
        settings = centering_sharpening_config({"loss": {"centering_sharpening": {
            "enabled": True, "center_momentum": center_momentum,
            "teacher_temperature": teacher_temperature, "student_temperature": student_temperature,
        }}})
        self.center_momentum = settings["center_momentum"]
        self.teacher_temperature = settings["teacher_temperature"]
        self.student_temperature = settings["student_temperature"]
        self.register_buffer("center", torch.zeros(feature_dim, dtype=torch.float32))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor,
                active: torch.Tensor) -> torch.Tensor:
        if (prediction.shape != target.shape or prediction.ndim != 3
                or active.shape != target.shape[:2] or active.dtype != torch.bool
                or target.shape[-1] != self.center.numel()):
            raise ValueError("Prediction, target, and validity shapes do not match")
        counts = active.sum(dim=1)
        if bool((counts == 0).any()):
            raise ValueError("Each target mask must contain at least one valid token")
        # Remove padded values before softmax, including non-finite padding.
        student = prediction.float().masked_fill(~active.unsqueeze(-1), 0.0)
        teacher = target.detach().float().masked_fill(~active.unsqueeze(-1), 0.0)
        probabilities = F.softmax((teacher - self.center) / self.teacher_temperature, dim=-1)
        log_probabilities = F.log_softmax(student / self.student_temperature, dim=-1)
        token_loss = -(probabilities * log_probabilities).sum(dim=-1)
        return (token_loss.masked_fill(~active, 0.0).sum(dim=1) / counts).mean()

    @torch.no_grad()
    def update_center(self, target: torch.Tensor, active: torch.Tensor) -> None:
        """Update once after a training step; never call during evaluation.

        Supply masked targets before replication for multiple context masks.
        All ranks contribute sums/counts so unequal valid lengths are weighted
        correctly. The current loss uses the previous step's running center.
        """
        values = target.detach().float()[active]
        statistics = torch.cat((values.sum(dim=0), values.new_tensor([len(values)])))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(statistics)
        if statistics[-1] <= 0:
            raise ValueError("Cannot update the center without valid target tokens")
        mean = statistics[:-1] / statistics[-1]
        self.center.mul_(self.center_momentum).add_(mean, alpha=1.0 - self.center_momentum)
