"""Trainable text and motion readouts over externally frozen token features."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from model.motion_transformer_1d import MotionTransformer1D
from model.specs import MODEL_SPECS
from model.modules import TransformerBlock1D, initialize_transformer
from model.pos_embs import ContinuousSinCosPosEmbed1D


@dataclass(frozen=True)
class AlignmentConfig:
    text_dim: int
    motion_dim: int
    embed_dim: int = 256
    depth: int = 6
    num_heads: int = 4
    ff_dim: int = 1024
    dropout: float = 0.1
    text_depth: int | None = None
    motion_depth: int | None = None
    text_pooling: str = "cls"
    motion_pooling: str = "cls"
    model_version: str = "v1"

    def __post_init__(self) -> None:
        if self.model_version != "v1":
            raise ValueError("AlignmentConfig requires model_version=v1")
        for name in ("text_dim", "motion_dim", "embed_dim", "depth", "num_heads", "ff_dim"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("text_pooling", "motion_pooling"):
            if getattr(self, name) not in {"cls", "mean"}:
                raise ValueError(f"{name} must be 'cls' or 'mean'")
        for name in ("text_depth", "motion_depth"):
            value = getattr(self, name)
            if value is None:
                value = self.depth
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            if value == 0 and getattr(self, name.replace("_depth", "_pooling")) != "mean":
                raise ValueError(f"{name}=0 requires mean pooling")
            object.__setattr__(self, name, value)
        if self.embed_dim % 2:
            raise ValueError("embed_dim must be even for sinusoidal positions")
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class _SequenceReadout(nn.Module):
    """Project token sequences and use CLS or valid-token mean readout."""

    def __init__(self, input_dim: int, config: AlignmentConfig, *, depth: int, pooling: str) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.pooling = pooling
        self.input_projection = nn.Linear(input_dim, config.embed_dim)
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.positions = ContinuousSinCosPosEmbed1D(config.embed_dim)
        self.blocks = nn.ModuleList(
            TransformerBlock1D(
                config.embed_dim,
                config.num_heads,
                mlp_ratio=config.ff_dim / config.embed_dim,
                drop=config.dropout,
                attn_drop=config.dropout,
            )
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.apply(initialize_transformer)
        if pooling == "cls":
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.input_dim:
            raise ValueError(f"Expected tokens [B,L,{self.input_dim}], got {tuple(tokens.shape)}")
        if tokens.shape[0] == 0 or tokens.shape[1] == 0:
            raise ValueError("Each sequence must contain at least one valid token")
        if not tokens.is_floating_point():
            raise ValueError("Token features must have a floating-point dtype")
        active = torch.as_tensor(valid_mask, device=tokens.device, dtype=torch.bool)
        if active.shape != tokens.shape[:2]:
            raise ValueError(f"valid_mask must have shape {tuple(tokens.shape[:2])}")
        if not active.any(dim=1).all():
            raise ValueError("Each sequence must contain at least one valid token")

        # Ignore even NaN/Inf cache padding before it enters the projection.
        cleaned = tokens.masked_fill(~active.unsqueeze(-1), 0)
        if not torch.isfinite(cleaned).all():
            raise ValueError("Valid token features must be finite")
        projected = self.input_projection(cleaned.to(self.input_projection.weight.dtype))
        if self.pooling == "cls":
            cls = self.cls_token.to(projected.dtype).expand(len(tokens), -1, -1)
            x = torch.cat((cls, projected), dim=1)
            active = torch.cat((torch.ones_like(active[:, :1]), active), dim=1)
        else:
            x = projected
        if self.blocks or self.pooling == "cls":
            positions = self.positions(torch.arange(x.shape[1], device=x.device))
            x = x + positions.to(dtype=x.dtype).unsqueeze(0)
        x = x.masked_fill(~active.unsqueeze(-1), 0)
        for block in self.blocks:
            x = block(x, active)
        if self.pooling == "cls":
            pooled = self.norm(x[:, 0])
        else:
            # Normalize token outputs before pooling so padding never contributes.
            x = self.norm(x).masked_fill(~active.unsqueeze(-1), 0)
            pooled = x.sum(dim=1) / active.sum(dim=1, keepdim=True).to(x.dtype)
        return F.normalize(pooled.float(), dim=-1)


class TextMotionAlignment(nn.Module):
    """Independent CLS Transformers aligned by a contrastive objective.

    Backbones are intentionally external: this module consumes raw motion or
    frozen JEPA motion tokens, and frozen text-backbone token sequences.
    """

    def __init__(self, config: AlignmentConfig) -> None:
        super().__init__()
        self.config = config
        self.text_encoder = _SequenceReadout(
            config.text_dim, config, depth=config.text_depth, pooling=config.text_pooling,
        )
        self.motion_encoder = _SequenceReadout(
            config.motion_dim, config, depth=config.motion_depth, pooling=config.motion_pooling,
        )

    def encode_text(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(tokens, valid_mask)

    def encode_motion(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return self.motion_encoder(tokens, valid_mask)

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




@dataclass(frozen=True)
class V2AlignmentConfig:
    """Resolved JEPA architecture and dataset timing, serialized in checkpoints."""

    text_dim: int
    motion_dim: int
    num_frames: int
    fps: float
    model_size: str = "base"
    model_version: str = "v2"
    embed_dim: int | None = None
    depth: int | None = None
    num_heads: int | None = None
    position_encoding: str = "rope"
    rope_theta: float = 100.0
    rope_time_scale: float = 1.0
    motion_adaptation: str = "scratch"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0

    def __post_init__(self):
        if self.motion_adaptation not in {"scratch", "lora"}:
            raise ValueError("motion_adaptation must be scratch or lora")
        if (isinstance(self.lora_rank, bool) or not isinstance(self.lora_rank, int)
                or self.lora_rank < 1 or not math.isfinite(self.lora_alpha) or self.lora_alpha <= 0
                or not math.isfinite(self.lora_dropout) or not 0 <= self.lora_dropout < 1):
            raise ValueError("Invalid LoRA rank, alpha, or dropout")
        if self.model_version != "v2" or self.model_size not in MODEL_SPECS:
            raise ValueError("Invalid TMR v2 version or model size")
        for name in ("text_dim", "motion_dim", "num_frames"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.fps, bool) or not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        object.__setattr__(self, "fps", float(self.fps))
        for name, expected in MODEL_SPECS[self.model_size].items():
            value = getattr(self, name)
            if value is not None and value != expected:
                raise ValueError(f"Saved {name} differs from JEPA size {self.model_size}")
            object.__setattr__(self, name, expected)
        if (self.position_encoding, self.rope_theta, self.rope_time_scale) != ("rope", 100.0, 1.0):
            raise ValueError("TMR v2 requires JEPA RoPE theta=100 and time_scale=1")


class TextMotionAlignmentV2(TextMotionAlignment):
    """Train a JEPA encoder from scratch against a frozen-text mean readout."""

    def __init__(self, config: V2AlignmentConfig):
        nn.Module.__init__(self)
        self.config = config
        text_config = AlignmentConfig(
            text_dim=config.text_dim, motion_dim=config.motion_dim,
            embed_dim=config.embed_dim, num_heads=config.num_heads,
            text_depth=0, text_pooling="mean",
        )
        self.text_encoder = _SequenceReadout(
            config.text_dim, text_config, depth=0, pooling="mean",
        )
        # Use JEPA's defaults for initialization, MLP, normalization and dropout.
        self.motion_encoder = MotionTransformer1D(
            in_chans=config.motion_dim, num_frames=config.num_frames,
            **MODEL_SPECS[config.model_size],
            position_encoding=config.position_encoding,
            rope_theta=config.rope_theta, rope_time_scale=config.rope_time_scale,
        )

    def encode_motion(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if (tokens.ndim != 3 or tokens.shape[-1] != self.config.motion_dim
                or not 0 < tokens.shape[1] <= self.config.num_frames or tokens.shape[0] == 0):
            raise ValueError("Motion must be a nonempty [B,T,motion_dim] within num_frames")
        if not tokens.is_floating_point():
            raise ValueError("Motion tokens must be floating point")
        active = torch.as_tensor(valid_mask, device=tokens.device, dtype=torch.bool)
        if active.shape != tokens.shape[:2] or not active.any(dim=1).all():
            raise ValueError("Every motion needs a matching mask and at least one valid frame")
        cleaned = tokens.masked_fill(~active.unsqueeze(-1), 0)
        if not torch.isfinite(cleaned).all():
            raise ValueError("Valid motion frames must be finite")
        padding = self.config.num_frames - tokens.shape[1]
        cleaned = F.pad(cleaned, (0, 0, 0, padding))
        active = F.pad(active, (0, padding), value=False)
        fps = torch.full(
            (tokens.shape[0],), self.config.fps, device=tokens.device, dtype=torch.float32,
        )
        encoded = self.motion_encoder(
            cleaned.to(self.motion_encoder.input_proj.weight.dtype),
            fps, masks=None, valid_frames=active,
        )
        pooled = encoded.float().sum(dim=1) / active.sum(dim=1, keepdim=True)
        return F.normalize(pooled, dim=-1)


def alignment_config_from_dict(values: dict) -> AlignmentConfig | V2AlignmentConfig:
    version = values.get("model_version", "v1")
    if version == "v1":
        return AlignmentConfig(**values)
    if version == "v2":
        return V2AlignmentConfig(**values)
    raise ValueError(f"Unsupported TMR model version: {version}")


def build_alignment(
    config: AlignmentConfig | V2AlignmentConfig, pretrained_encoder: nn.Module | None = None,
) -> TextMotionAlignment:
    if isinstance(config, V2AlignmentConfig) and config.motion_adaptation == "lora":
        from .lora import TextMotionAlignmentLoRA
        return TextMotionAlignmentLoRA(config, pretrained_encoder=pretrained_encoder)
    if pretrained_encoder is not None:
        raise ValueError("Pretrained initialization requires LoRA adaptation")
    return TextMotionAlignmentV2(config) if isinstance(config, V2AlignmentConfig) else TextMotionAlignment(config)


def parameter_counts(model: TextMotionAlignment) -> dict[str, int]:
    counts = {
        name: sum(p.numel() for p in branch.parameters() if p.requires_grad)
        for name, branch in (("motion", model.motion_encoder), ("text", model.text_encoder))
    }
    result = {**counts, "total": sum(counts.values())}
    if getattr(model.config, "motion_adaptation", None) == "lora":
        result["frozen_motion"] = sum(p.numel() for p in model.motion_encoder.parameters() if not p.requires_grad)
    return result


__all__ = [
    "AlignmentConfig", "TextMotionAlignment", "V2AlignmentConfig",
    "TextMotionAlignmentV2", "alignment_config_from_dict", "build_alignment", "parameter_counts",
]
