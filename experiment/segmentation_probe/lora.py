"""Experimental LoRA encoder plus a token-local linear segmentation head.

Only adapters and the head are optimized. Pretrained parameters (including
biases and LayerNorms) remain frozen. The default targets adapt the full fused
Q/K/V projection and attention output projection in every transformer block.
A is Kaiming-initialized and B is zero, so insertion preserves encoder outputs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .model import FrameLinearProbe

DEFAULT_TARGETS = ("attn.qkv", "attn.proj")


def _validate_options(rank: int, alpha: float, dropout: float) -> None:
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("LoRA rank must be a positive integer")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("LoRA alpha must be finite and positive")
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("LoRA dropout must be in [0, 1)")


class LoRALinear(nn.Module):
    """Frozen affine map plus (alpha / rank) * B(A(dropout(x)))."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        _validate_options(rank, alpha, dropout)
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear requires an nn.Linear base")
        if rank > min(base.in_features, base.out_features):
            raise ValueError("LoRA rank exceeds the projection dimensions")
        self.base = base.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.rank, self.alpha = rank, float(alpha)
        self.scaling = self.alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The frozen base still participates in autograd for earlier adapters.
        update = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return self.base(x) + update * self.scaling


def inject_lora(encoder: nn.Module, *, rank: int = 8, alpha: float = 16.0,
                dropout: float = 0.0,
                targets: Sequence[str] = DEFAULT_TARGETS) -> tuple[str, ...]:
    """Freeze an encoder and wrap matching Linear modules, in place.

    Targets are exact module names or dotted suffixes, e.g. attn.qkv,
    mlp.fc1, or blocks.0.attn.proj. Validate every target before mutation.
    """
    _validate_options(rank, alpha, dropout)
    if isinstance(targets, str):
        raise TypeError("targets must be a sequence of module names")
    targets = tuple(targets)
    if not targets or any(not isinstance(t, str) or not t.strip(".") for t in targets):
        raise ValueError("At least one nonempty LoRA target is required")
    if len(set(targets)) != len(targets):
        raise ValueError("Duplicate LoRA targets")
    if any(isinstance(module, LoRALinear) for module in encoder.modules()):
        raise ValueError("Encoder already contains LoRA adapters")
    matches, found = [], set()
    for name, module in encoder.named_modules():
        matched = [target for target in targets
                   if name == target or name.endswith("." + target)]
        if not matched:
            continue
        if not isinstance(module, nn.Linear):
            raise ValueError(f"LoRA target {name!r} is not an nn.Linear")
        if rank > min(module.in_features, module.out_features):
            raise ValueError(f"LoRA rank exceeds projection dimensions: {name}")
        found.update(matched)
        matches.append((name, module))
    missing = set(targets) - found
    if missing:
        raise ValueError(f"Unmatched LoRA targets: {sorted(missing)}")
    encoder.requires_grad_(False)
    for parameter in encoder.parameters():
        parameter.grad = None
    for name, module in matches:
        parent_name, _, child_name = name.rpartition(".")
        parent = encoder.get_submodule(parent_name) if parent_name else encoder
        setattr(parent, child_name, LoRALinear(module, rank, alpha, dropout))
    return tuple(name for name, _ in matches)


class LoRASegmentationProbe(nn.Module):
    """Train adapters and FrameLinearProbe with fixed train-token scaling.

    Backbone dropout/stochastic depth stay disabled. Only LoRA dropout follows
    train/eval mode. This is parameter-efficient adaptation with a linear head;
    encoder features change during training and cannot use frozen token caches.
    """

    def __init__(self, encoder: nn.Module, *, rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0, targets: Sequence[str] = DEFAULT_TARGETS,
                 num_classes: int = 120):
        super().__init__()
        if encoder.token_layout.kind != "1d":
            raise ValueError("LoRA segmentation currently supports only 1D encoders")
        self.adapter_names = inject_lora(
            encoder, rank=rank, alpha=alpha, dropout=dropout, targets=targets)
        self.encoder = encoder
        reference = next(encoder.parameters())
        self.head = FrameLinearProbe(
            int(encoder.embed_dim), encoder.token_layout.temporal_patch_size,
            num_classes).to(device=reference.device, dtype=torch.float32)
        self.register_buffer("feature_mean", torch.zeros(
            int(encoder.embed_dim), device=reference.device))
        self.register_buffer("feature_scale", torch.ones(
            int(encoder.embed_dim), device=reference.device))
        self.train()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        for module in self.encoder.modules():
            if isinstance(module, LoRALinear):
                module.train(mode)
        return self

    def adapter_parameters(self):
        for module in self.encoder.modules():
            if isinstance(module, LoRALinear):
                yield module.lora_A
                yield module.lora_B

    def forward(self, motion: torch.Tensor, fps: torch.Tensor,
                valid_frames: torch.Tensor, *, use_bfloat16: bool = False):
        with torch.autocast(device_type=motion.device.type, dtype=torch.bfloat16,
                            enabled=use_bfloat16):
            tokens = self.encoder(motion, fps, valid_frames=valid_frames)
        # Casting retains the gradient path into the adapters.
        with torch.autocast(device_type=motion.device.type, enabled=False):
            tokens = (tokens.float() - self.feature_mean) / self.feature_scale
            return self.head(tokens)

    def adaptation_state_dict(self) -> dict[str, torch.Tensor]:
        """Compact state; restoring also requires the original pretrained model."""
        state = {name: parameter.detach().cpu().clone()
                 for name, parameter in self.named_parameters() if parameter.requires_grad}
        state.update(feature_mean=self.feature_mean.detach().cpu().clone(),
                     feature_scale=self.feature_scale.detach().cpu().clone())
        return state

    def load_adaptation_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        expected = {name: parameter for name, parameter in self.named_parameters()
                    if parameter.requires_grad}
        expected.update(feature_mean=self.feature_mean, feature_scale=self.feature_scale)
        if state.keys() != expected.keys():
            raise ValueError("Adapter/head checkpoint keys do not match this model")
        for name, value in state.items():
            if (not isinstance(value, torch.Tensor) or value.shape != expected[name].shape
                    or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid adapter/head tensor: {name}")
        if (state["feature_scale"] <= 0).any():
            raise ValueError("Feature scales must be positive")
        with torch.no_grad():
            for name, destination in expected.items():
                destination.copy_(state[name])
