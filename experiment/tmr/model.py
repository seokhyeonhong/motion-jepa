"""Trainable RoPE-based JEPA encoders for text-motion retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from model.motion_transformer_1d import MotionTransformer1D
from model.modules import initialize_transformer
from model.specs import MODEL_SPECS


@dataclass(frozen=True)
class AlignmentConfig:
    """Shared JEPA size preset and modality timing, saved in checkpoints."""

    text_dim: int
    motion_dim: int
    num_frames: int
    fps: float
    model_size: str = "base"
    text_num_tokens: int = 1
    embed_dim: int | None = None
    depth: int | None = None
    num_heads: int | None = None
    retrieval_dim: int = 256
    motion_encoder: str = "transformer"
    probe_heads: int = 6
    patchify: bool = False
    motion_depth: int | None = None

    def __post_init__(self):
        if self.motion_depth is not None:
            if isinstance(self.motion_depth, bool) or not isinstance(self.motion_depth, int) or self.motion_depth < 1:
                raise ValueError("motion_depth must be a positive integer")
            if self.motion_encoder == "attentive_probe":
                raise ValueError("motion_depth requires a Transformer motion encoder")
        if self.patchify and (self.motion_encoder != "transformer" or self.num_frames < 3):
            raise ValueError("patchify requires a raw-motion transformer with at least three frames")
        if self.motion_encoder not in {"transformer", "attentive_probe", "transformer_probe", "mlp_probe"}:
            raise ValueError("Invalid motion encoder")
        if self.probe_heads < 1 or (self.motion_encoder == "attentive_probe" and self.motion_dim % self.probe_heads):
            raise ValueError("probe_heads must divide motion feature dimension")
        if self.model_size not in MODEL_SPECS:
            raise ValueError("Invalid model size")
        for name in ("text_dim", "motion_dim", "num_frames", "text_num_tokens", "retrieval_dim"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.fps, bool) or not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        object.__setattr__(self, "fps", float(self.fps))
        for name, expected in MODEL_SPECS[self.model_size].items():
            value = getattr(self, name)
            if value is not None and value != expected:
                raise ValueError(f"Saved {name} differs from size {self.model_size}")
            object.__setattr__(self, name, expected)
        if self.retrieval_dim != 256:
            raise ValueError("retrieval_dim must be 256")


class SequenceEncoder(nn.Module):
    """Shared MotionJEPA backbone followed by a masked retrieval readout."""

    def __init__(self, input_dim: int, config: AlignmentConfig, *, num_tokens: int, fps: float, patchify: bool = False, depth: int | None = None):
        super().__init__()
        self.input_dim = input_dim
        self.fps = fps
        spec = dict(MODEL_SPECS[config.model_size])
        if depth is not None:
            spec["depth"] = depth
        self.backbone = MotionTransformer1D(
            in_chans=input_dim, num_frames=num_tokens, patchify=patchify, **spec,
        )
        self.output_projection = (
            nn.Identity() if config.embed_dim == config.retrieval_dim
            else nn.Linear(config.embed_dim, config.retrieval_dim)
        )
        self.output_projection.apply(initialize_transformer)

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if (tokens.ndim != 3 or tokens.shape[-1] != self.input_dim
                or tokens.shape[0] == 0 or not 0 < tokens.shape[1] <= self.backbone.num_frames):
            raise ValueError(
                f"Expected nonempty tokens [B,L,{self.input_dim}] with L <= {self.backbone.num_frames}"
            )
        if not tokens.is_floating_point():
            raise ValueError("Token features must have a floating-point dtype")
        active = torch.as_tensor(valid_mask, device=tokens.device, dtype=torch.bool)
        if active.shape != tokens.shape[:2] or not active.any(dim=1).all():
            raise ValueError("Each sequence needs a matching mask and at least one valid token")
        cleaned = tokens.masked_fill(~active.unsqueeze(-1), 0)
        if not torch.isfinite(cleaned).all():
            raise ValueError("Valid token features must be finite")
        padding = self.backbone.num_frames - tokens.shape[1]
        cleaned = F.pad(cleaned, (0, 0, 0, padding))
        active = F.pad(active, (0, padding), value=False)
        token_active = self.backbone.token_layout.valid_token_mask(active)
        if not token_active.any(dim=1).all():
            raise ValueError("Each sequence needs at least one complete motion patch")
        rates = torch.full(
            (len(tokens),), self.fps, device=tokens.device, dtype=torch.float32,
        )
        encoded = self.backbone(
            cleaned.to(self.backbone.input_proj.weight.dtype), rates, valid_frames=active,
        )
        pooled = encoded.sum(dim=1) / token_active.sum(dim=1, keepdim=True).to(encoded.dtype)
        return F.normalize(self.output_projection(pooled).float(), dim=-1)


class MLPProbe(nn.Module):
    """Token-wise GELU MLP, masked mean pooling, and retrieval projection."""

    def __init__(self, config: AlignmentConfig):
        super().__init__()
        self.input_dim = config.motion_dim
        self.num_frames = config.num_frames
        width = MODEL_SPECS[config.model_size]["embed_dim"]
        depth = config.motion_depth if config.motion_depth is not None else 2
        layers = []
        for index in range(depth):
            layers.extend([nn.Linear(config.motion_dim if index == 0 else width, width), nn.GELU()])
        self.layers = nn.Sequential(*layers)
        self.output_projection = nn.Linear(width, config.retrieval_dim)
        self.apply(initialize_transformer)

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if (tokens.ndim != 3 or tokens.shape[-1] != self.input_dim
                or tokens.shape[0] == 0 or not 0 < tokens.shape[1] <= self.num_frames):
            raise ValueError("Expected nonempty motion tokens with matching feature width")
        if not tokens.is_floating_point():
            raise ValueError("Token features must have a floating-point dtype")
        active = torch.as_tensor(valid_mask, device=tokens.device, dtype=torch.bool)
        if active.shape != tokens.shape[:2] or not active.any(dim=1).all():
            raise ValueError("Each sequence needs a matching mask and at least one valid token")
        cleaned = tokens.masked_fill(~active.unsqueeze(-1), 0)
        if not torch.isfinite(cleaned).all():
            raise ValueError("Valid token features must be finite")
        encoded = self.layers(cleaned.to(self.layers[0].weight.dtype))
        encoded = encoded.masked_fill(~active.unsqueeze(-1), 0)
        pooled = encoded.sum(dim=1) / active.sum(dim=1, keepdim=True).to(encoded.dtype)
        return F.normalize(self.output_projection(pooled).float(), dim=-1)


class TextMotionAlignment(nn.Module):
    """Independent RoPE-based JEPA text and motion encoders for retrieval."""

    def __init__(self, config: AlignmentConfig, *, motion_mean=None, motion_std=None):
        super().__init__()
        self.config = config
        self.text_encoder = SequenceEncoder(
            config.text_dim, config, num_tokens=config.text_num_tokens, fps=1.0,
        )
        if config.motion_encoder == "attentive_probe":
            from experiment.probing import AttentiveProbe
            mean = torch.zeros(config.motion_dim) if motion_mean is None else motion_mean
            std = torch.ones(config.motion_dim) if motion_std is None else motion_std
            self.motion_encoder = AttentiveProbe(
                config.motion_dim, config.retrieval_dim, mean, std,
                num_heads=config.probe_heads, pool=True,
            )
        elif config.motion_encoder == "mlp_probe":
            self.motion_encoder = MLPProbe(config)
        else:
            self.motion_encoder = SequenceEncoder(
                config.motion_dim, config, num_tokens=config.num_frames, fps=config.fps,
                patchify=config.patchify, depth=config.motion_depth,
            )

        if config.motion_encoder in {"transformer_probe", "mlp_probe"}:
            mean = torch.zeros(config.motion_dim) if motion_mean is None else torch.as_tensor(motion_mean).float()
            std = torch.ones(config.motion_dim) if motion_std is None else torch.as_tensor(motion_std).float()
            if mean.shape != (config.motion_dim,) or std.shape != mean.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
                raise ValueError("Invalid frozen-feature normalization")
            self.register_buffer("feature_mean", mean.clone())
            self.register_buffer("feature_std", std.clone())

    def encode_text(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(tokens, valid_mask)

    def encode_motion(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if self.config.motion_encoder in {"transformer_probe", "mlp_probe"}:
            tokens = (tokens - self.feature_mean) / self.feature_std
        values = self.motion_encoder(tokens, valid_mask)
        return F.normalize(values.float(), dim=-1) if self.config.motion_encoder == "attentive_probe" else values

    def forward(
        self,
        motion_tokens: torch.Tensor,
        motion_mask: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if motion_tokens.shape[0] != text_tokens.shape[0]:
            raise ValueError("Text and motion batch sizes must match")
        return self.encode_motion(motion_tokens, motion_mask), self.encode_text(text_tokens, text_mask)


def parameter_counts(model: TextMotionAlignment) -> dict[str, int]:
    counts = {
        name: sum(p.numel() for p in branch.parameters() if p.requires_grad)
        for name, branch in (("motion", model.motion_encoder), ("text", model.text_encoder))
    }
    return {**counts, "total": sum(counts.values())}


__all__ = ["AlignmentConfig", "SequenceEncoder", "TextMotionAlignment", "parameter_counts"]
