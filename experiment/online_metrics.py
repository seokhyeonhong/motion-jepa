"""Fixed-validation representation and prediction diagnostics for frame-token JEPA.

Geometry uses valid-frame mean-pooled EMA embeddings. Prediction compares the
online encoder/predictor with the EMA target, matching the training target.
This evaluator writes no files and owns a mask stream separate from training.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset.motion_dataset import MotionDataset
from mask.utils import apply_index_masks, index_mask_validity, repeat_mask_blocks


_EPS = 1.0e-12


def effective_rank(values: torch.Tensor) -> float:
    """RankMe effective rank from the singular values of an uncentered matrix."""
    matrix = torch.as_tensor(values, dtype=torch.float64, device="cpu")
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError(f"RankMe expects a non-empty matrix, got {matrix.shape}")
    singular = torch.linalg.svdvals(matrix)
    total = singular.sum()
    if float(total) <= _EPS:
        return 1.0
    probabilities = singular / total
    probabilities = probabilities[probabilities > 0]
    return float(torch.exp(-(probabilities * probabilities.log()).sum()))


def mean_off_diagonal_cosine(values: torch.Tensor) -> float:
    matrix = torch.as_tensor(values, dtype=torch.float64, device="cpu")
    if matrix.ndim != 2 or len(matrix) < 2:
        return 0.0
    normalized = F.normalize(matrix, dim=-1, eps=_EPS)
    total = normalized.sum(dim=0).square().sum() - normalized.square().sum()
    return float(total / (len(matrix) * (len(matrix) - 1)))


def covariance_metrics(values: torch.Tensor) -> dict[str, float]:
    matrix = torch.as_tensor(values, dtype=torch.float64, device="cpu")
    if matrix.ndim != 2 or len(matrix) < 2:
        return {
            "largest_eigenvalue_ratio": 1.0,
            "effective_rank": 1.0,
            "condition_number": 1.0,
            "off_diagonal_abs_mean": 0.0,
        }
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, len(matrix) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total = eigenvalues.sum()
    if float(total) <= _EPS:
        largest_ratio = rank = condition = 1.0
    else:
        probabilities = eigenvalues / total
        positive = probabilities[probabilities > 0]
        largest_ratio = float(probabilities[-1])
        rank = float(torch.exp(-(positive * positive.log()).sum()))
        floor = max(_EPS, float(eigenvalues[-1]) * _EPS)
        condition = float(eigenvalues[-1] / eigenvalues[0].clamp_min(floor))
    if covariance.shape[0] < 2:
        off_diagonal = 0.0
    else:
        off_diagonal = float(
            (covariance.abs().sum() - covariance.diagonal().abs().sum())
            / (covariance.numel() - covariance.shape[0])
        )
    return {
        "largest_eigenvalue_ratio": largest_ratio,
        "effective_rank": rank,
        "condition_number": condition,
        "off_diagonal_abs_mean": off_diagonal,
    }


def feature_matrix_metrics(values: torch.Tensor) -> dict[str, float]:
    matrix = torch.as_tensor(values, dtype=torch.float64, device="cpu")
    return {
        "rankme": effective_rank(matrix),
        "mean_std": float(matrix.std(dim=0, correction=0).mean()),
        "mean_off_diagonal_cosine": mean_off_diagonal_cosine(matrix),
        "covariance": covariance_metrics(matrix),
    }


def prediction_gain(mse: float, baseline_mse: float) -> float:
    if baseline_mse <= _EPS:
        return 0.0 if mse <= _EPS else 1.0 - mse / _EPS
    return 1.0 - mse / baseline_mse


@dataclass
class _PredictionAccumulator:
    count: int = 0
    squared_error: float = 0.0
    smooth_l1: float = 0.0
    cosine: float = 0.0

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        selected: torch.Tensor | None = None,
    ) -> None:
        prediction = prediction.detach().float().cpu()
        target = target.detach().float().cpu()
        if selected is not None:
            selected = selected.to(dtype=torch.bool, device="cpu")
            prediction = prediction[selected]
            target = target[selected]
        if prediction.numel() == 0:
            return
        if prediction.ndim != 2:
            prediction = prediction.reshape(-1, prediction.shape[-1])
            target = target.reshape(-1, target.shape[-1])
        self.count += len(prediction)
        self.squared_error += float((prediction - target).square().mean(dim=-1).sum())
        self.smooth_l1 += float(
            F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1).sum()
        )
        self.cosine += float(F.cosine_similarity(prediction, target, dim=-1).sum())

    def summary(self) -> dict[str, float]:
        denominator = max(1, self.count)
        return {
            "count": int(self.count),
            "mse": self.squared_error / denominator,
            "smooth_l1": self.smooth_l1 / denominator,
            "cosine_similarity": self.cosine / denominator,
        }


class _TargetMoments:
    def __init__(self) -> None:
        self.count = 0
        self.sum: torch.Tensor | None = None
        self.square_sum = 0.0

    def update(self, target: torch.Tensor, selected: torch.Tensor | None = None) -> None:
        values = target.detach().to(device="cpu", dtype=torch.float64)
        if selected is not None:
            values = values[selected.to(dtype=torch.bool, device="cpu")]
        values = values.reshape(-1, values.shape[-1])
        if not len(values):
            return
        self.count += len(values)
        batch_sum = values.sum(dim=0)
        self.sum = batch_sum if self.sum is None else self.sum + batch_sum
        self.square_sum += float(values.square().sum())

    def baseline_mse(self) -> float:
        if self.count == 0 or self.sum is None:
            return 0.0
        total_variation = self.square_sum - float(self.sum.square().sum()) / self.count
        return max(0.0, total_variation / (self.count * len(self.sum)))


def tensorboard_metrics(summary: dict) -> dict[str, float]:
    """Emit changing diagnostics, keeping fixed counts/metadata out of curves."""
    excluded = {
        "heldout_jepa/count",
        "num_samples",
        "feature_dim",
        "rankme_ceiling",
        "covariance_rank_ceiling",
        "representation/standardized/mean_std",
    }
    output = {}
    def collect(path, value):
        if isinstance(value, dict):
            for key, child in value.items():
                collect(f"{path}/{key}" if path else key, child)
        elif isinstance(value, (int, float)):
            if not math.isfinite(value):
                raise ValueError(f"Non-finite online metric: {path}={value}")
            if path not in excluded:
                output[path] = float(value)
    collect("", summary)
    return output


def evaluation_interval(metric_config: dict) -> int:
    if not metric_config.get("enabled", False):
        return 0
    value = metric_config.get("interval", 1000)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("online_metrics.interval must be a positive integer (iterations)")
    return value


class OnlineRepresentationMetrics:
    """Evaluate the same unlabeled clips and masks on every call."""

    def __init__(self, training_config: dict, metric_config: dict, *, device, collator):
        self.device = torch.device(device)
        self.batch_size = int(metric_config.get("batch_size", 64))
        self.num_workers = int(metric_config.get("num_workers", 0))
        self.num_samples = int(metric_config.get("num_samples", 1024))
        self.seed = int(metric_config.get("seed", 42))
        if self.batch_size <= 0 or self.num_samples < 2 or self.num_workers < 0:
            raise ValueError("online_metrics requires positive batch size, >=2 samples, and >=0 workers")
        data = training_config["data"]
        self.dataset = MotionDataset(
            root_path=Path(data["root_path"]),
            meta_files=[str(metric_config.get("meta_file", "val.txt"))],
            num_frames=int(data["num_frames"]), fps=int(data["fps"]),
            motion_dim=int(data["motion_dim"]),
            representation=data.get("representation", "MotionJEPAMotionRep"),
            normalize=bool(data.get("normalize", False)), stats_path=data.get("stats_path"),
        )
        count = min(self.num_samples, len(self.dataset))
        if count < 2:
            raise ValueError("Online metrics need at least two validation clips")
        self.indices = torch.randperm(len(self.dataset), generator=torch.Generator().manual_seed(self.seed))[:count].tolist()
        self.collator = collator
        self.initial_collator_state = collator.state_dict()
        self.initial_collator_state["counter"] = self.seed - 1
        meta = training_config["meta"]
        self.amp_dtype = (torch.bfloat16 if meta.get("use_bfloat16", False)
                          else torch.float16 if meta.get("use_float16", False) else None)

    def _loader(self):
        self.collator.load_state_dict(self.initial_collator_state)
        return DataLoader(
            Subset(self.dataset, self.indices), batch_size=self.batch_size,
            shuffle=False, drop_last=False, num_workers=self.num_workers,
            generator=torch.Generator().manual_seed(self.seed),
        )

    @torch.no_grad()
    def evaluate(self, encoder, predictor, target_encoder, *, centering_sharpening=None) -> dict[str, Any]:
        if any(model.training for model in (encoder, predictor, target_encoder)):
            raise ValueError("Online metrics require eval-mode models")
        if encoder.token_layout.kind != "1d":
            raise ValueError("Online metrics require 1D models")
        started = time.perf_counter()
        pooled = []
        temporal_std = temporal_variance = objective_sum = 0.0
        objective_count = 0
        cross_entropy_sum = 0.0
        overall = _PredictionAccumulator()
        target_moments = _TargetMoments()
        for motion, fps, lengths in self._loader():
            token_lengths = encoder.token_layout.valid_token_lengths(lengths)
            masks_enc, masks_pred = self.collator.sample_masks(token_lengths.tolist())
            motion = motion.to(self.device, dtype=torch.float32)
            fps = fps.to(self.device, dtype=torch.float32)
            valid = torch.arange(motion.shape[1], device=self.device)[None] < lengths.to(self.device)[:, None]
            masks_enc = [mask.to(self.device) for mask in masks_enc]
            masks_pred = [mask.to(self.device) for mask in masks_pred]
            amp = torch.autocast(self.device.type, dtype=self.amp_dtype) if self.amp_dtype else nullcontext()
            with amp:
                full = target_encoder(motion, fps, valid_frames=valid)
                target = full if centering_sharpening is not None else F.layer_norm(full, (full.shape[-1],))
                target = apply_index_masks(target, masks_pred)
                target = repeat_mask_blocks(target, len(motion), len(masks_enc))
                context = encoder(motion, fps, masks_enc, valid_frames=valid)
                prediction = predictor(context, fps, masks_enc, masks_pred)
            for features, length in zip(full.float().cpu(), token_lengths.tolist()):
                features = features[:length]
                pooled.append(features.mean(0))
                temporal_std += float(features.std(0, correction=0).mean())
                temporal_variance += float(features.var(0, correction=0).mean())
            active = repeat_mask_blocks(index_mask_validity(masks_pred), len(motion), len(masks_enc))
            overall.update(prediction, target, active)
            target_moments.update(target, active)
            # Average valid targets per mask/sample, then average rows, as in training.
            error = F.smooth_l1_loss(prediction.float(), target.float(), reduction="none").mean(-1)
            row_loss = error.masked_fill(~active, 0).sum(1) / active.sum(1)
            objective_sum += float(row_loss.sum())
            objective_count += len(row_loss)
            if centering_sharpening is not None:
                cross_entropy_sum += float(centering_sharpening(prediction, target, active)) * len(row_loss)
        matrix = torch.stack(pooled).double()
        if not torch.isfinite(matrix).all():
            raise ValueError("Non-finite validation embeddings")
        std = matrix.std(0, correction=0)
        standardized = (matrix - matrix.mean(0)) / std.clamp_min(1e-6)
        heldout = overall.summary()
        heldout["smooth_l1"] = objective_sum / objective_count
        if centering_sharpening is not None:
            heldout["cross_entropy"] = cross_entropy_sum / objective_count
        heldout["trivial_mean_mse"] = target_moments.baseline_mse()
        heldout["prediction_gain"] = prediction_gain(heldout["mse"], heldout["trivial_mean_mse"])
        result = {
            "representation": {
                "raw": feature_matrix_metrics(matrix),
                "standardized": feature_matrix_metrics(standardized),
                "temporal_mean_std": temporal_std / len(matrix),
                "temporal_mean_variance": temporal_variance / len(matrix),
                "near_zero_channels": int((std < 1e-6).sum()),
            },
            "heldout_jepa": heldout,
            "num_samples": len(matrix), "feature_dim": matrix.shape[1],
            "rankme_ceiling": min(matrix.shape),
            "covariance_rank_ceiling": min(len(matrix) - 1, matrix.shape[1]),
            "elapsed_seconds": time.perf_counter() - started,
        }
        tensorboard_metrics(result)  # Reject invalid metrics before emitting any curves.
        return result


__all__ = ["OnlineRepresentationMetrics", "effective_rank", "covariance_metrics",
           "mean_off_diagonal_cosine", "feature_matrix_metrics", "prediction_gain",
           "tensorboard_metrics", "evaluation_interval"]
