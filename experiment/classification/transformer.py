"""Raw-motion classifier using the shared JEPA-scaled frame encoder."""

from __future__ import annotations

import torch
from torch import nn

from model.motion_transformer_1d import MotionTransformer1D
from model.modules import initialize_transformer
from model.specs import MODEL_SPECS


class MotionTransformerClassifier(nn.Module):
    """Train a RoPE frame encoder and masked-mean classification head from scratch."""

    def __init__(
        self,
        motion_dim: int | None = None,
        num_frames: int = 150,
        num_classes: int = 100,
        *,
        input_dim: int | None = None,
        model_size: str = "base",
    ) -> None:
        super().__init__()
        if model_size not in MODEL_SPECS:
            raise ValueError(f"Unknown model size: {model_size}")
        if motion_dim is not None and input_dim is not None and motion_dim != input_dim:
            raise ValueError("motion_dim and input_dim must match when both are provided")
        resolved_dim = input_dim if input_dim is not None else motion_dim if motion_dim is not None else 366
        for name, value in (("input_dim", resolved_dim), ("num_frames", num_frames), ("num_classes", num_classes)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.input_dim = self.motion_dim = resolved_dim
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.model_size = model_size
        self.embed_dim = MODEL_SPECS[model_size]["embed_dim"]
        self.config = {
            "model_version": "raw_rope_mean_v1",
            "model_size": model_size,
            **MODEL_SPECS[model_size],
            "input_dim": self.input_dim,
            "num_frames": num_frames,
            "num_classes": num_classes,
            "pooling": "valid_frame_mean",
            "position_encoding": "rope",
            "rope_theta": 100.0,
            "rope_time_scale": 1.0,
            "mlp_ratio": 4.0,
            "qkv_bias": True,
            "norm_eps": 1e-6,
            "dropout": 0.0,
            "drop_path_rate": 0.0,
            "pretrained": False,
        }
        self.encoder = MotionTransformer1D(
            in_chans=self.input_dim, num_frames=num_frames, **MODEL_SPECS[model_size],
        )
        self.head = nn.Linear(self.embed_dim, num_classes)
        self.head.apply(initialize_transformer)

    def forward(
        self,
        motion: torch.Tensor,
        valid_frames: torch.Tensor,
        fps: torch.Tensor,
    ) -> torch.Tensor:
        if motion.ndim != 3 or motion.shape[1:] != (self.num_frames, self.input_dim):
            raise ValueError(f"Expected motion [B,{self.num_frames},{self.input_dim}], got {tuple(motion.shape)}")
        if valid_frames.dtype != torch.bool or valid_frames.shape != motion.shape[:2]:
            raise ValueError("Expected a boolean validity mask matching the motion frames")
        active = valid_frames.to(device=motion.device)
        if not active.any(dim=1).all():
            raise ValueError("Each sample must contain a valid frame")
        fps = torch.as_tensor(fps, device=motion.device, dtype=torch.float32)
        if fps.shape != (motion.shape[0],) or not torch.isfinite(fps).all() or not (fps > 0).all():
            raise ValueError("Expected finite positive FPS per sample")
        cleaned = motion.masked_fill(~active.unsqueeze(-1), 0)
        tokens = self.encoder(cleaned, fps, valid_frames=active)
        pooled = tokens.float().masked_fill(~active.unsqueeze(-1), 0).sum(dim=1)
        pooled = pooled / active.sum(dim=1, keepdim=True)
        return self.head(pooled.to(tokens.dtype))

    def parameter_counts(self) -> dict[str, int]:
        encoder = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        head = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        return {"encoder": encoder, "head": head, "total": encoder + head}


__all__ = ["MotionTransformerClassifier"]
