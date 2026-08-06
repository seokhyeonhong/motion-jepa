"""Shared transformer building blocks for Motion-JEPA models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * random.floor() / keep


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(self.activation(self.fc1(x)))))


class SelfAttention(nn.Module):
    """Multi-head self-attention with an optional active-key mask."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"Embedding dimension {dim} is not divisible by {num_heads} heads")
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = float(attn_drop)

    def forward(
        self,
        x: torch.Tensor,
        active: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention_mask = None
        if active is not None:
            active = active.to(device=x.device, dtype=torch.bool)
            if active.shape != (batch, tokens):
                raise ValueError(
                    f"Active mask must have shape {(batch, tokens)}, got {tuple(active.shape)}"
                )
            safe_active = active.clone()
            empty = ~safe_active.any(dim=1)
            if empty.any():
                safe_active[empty, 0] = True
            attention_mask = safe_active[:, None, None, :]
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj_drop(self.proj(output))


class TransformerBlock1D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.attn_norm = norm_layer(dim)
        self.attn = SelfAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.mlp_norm = norm_layer(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        active: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.attn_norm(x), active))
        if active is not None:
            x = x * active.unsqueeze(-1).to(dtype=x.dtype)
        x = x + self.drop_path(self.mlp(self.mlp_norm(x)))
        if active is not None:
            x = x * active.unsqueeze(-1).to(dtype=x.dtype)
        return x


class AxialBlock2D(nn.Module):
    """Temporal-then-spatial attention over a masked ``[B,T,J,D]`` grid."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.temporal_norm = norm_layer(dim)
        self.temporal_attn = SelfAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.spatial_norm = norm_layer(dim)
        self.spatial_attn = SelfAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.mlp_norm = norm_layer(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    @staticmethod
    def _zero_inactive(x: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        return x * active.unsqueeze(-1).to(dtype=x.dtype)

    def forward(self, x: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        batch, frames, joints, dim = x.shape
        temporal = self.temporal_norm(x).permute(0, 2, 1, 3).reshape(
            batch * joints, frames, dim
        )
        temporal_active = active.permute(0, 2, 1).reshape(batch * joints, frames)
        temporal = self.temporal_attn(temporal, temporal_active).reshape(
            batch, joints, frames, dim
        ).permute(0, 2, 1, 3)
        x = self._zero_inactive(x + self.drop_path(temporal), active)

        spatial = self.spatial_norm(x).reshape(batch * frames, joints, dim)
        spatial_active = active.reshape(batch * frames, joints)
        spatial = self.spatial_attn(spatial, spatial_active).reshape(
            batch, frames, joints, dim
        )
        x = self._zero_inactive(x + self.drop_path(spatial), active)
        return self._zero_inactive(
            x + self.drop_path(self.mlp(self.mlp_norm(x))), active
        )


class PredictorAxialBlock2D(nn.Module):
    """Axial attention over separate context and target-query streams."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.temporal_norm = norm_layer(dim)
        self.temporal_attn = SelfAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.spatial_norm = norm_layer(dim)
        self.spatial_attn = SelfAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.mlp_norm = norm_layer(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    @staticmethod
    def _zero_inactive(x: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        return x * active.unsqueeze(-1).to(dtype=x.dtype)

    def forward(self, x: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        batch, streams, frames, joints, dim = x.shape
        temporal = self.temporal_norm(x).permute(0, 3, 1, 2, 4).reshape(
            batch * joints, streams * frames, dim
        )
        temporal_active = active.permute(0, 3, 1, 2).reshape(
            batch * joints, streams * frames
        )
        temporal = self.temporal_attn(temporal, temporal_active).reshape(
            batch, joints, streams, frames, dim
        ).permute(0, 2, 3, 1, 4)
        x = self._zero_inactive(x + self.drop_path(temporal), active)

        spatial = self.spatial_norm(x).permute(0, 2, 1, 3, 4).reshape(
            batch * frames, streams * joints, dim
        )
        spatial_active = active.permute(0, 2, 1, 3).reshape(
            batch * frames, streams * joints
        )
        spatial = self.spatial_attn(spatial, spatial_active).reshape(
            batch, frames, streams, joints, dim
        ).permute(0, 2, 1, 3, 4)
        x = self._zero_inactive(x + self.drop_path(spatial), active)
        return self._zero_inactive(
            x + self.drop_path(self.mlp(self.mlp_norm(x))), active
        )


def initialize_transformer(module: nn.Module, std: float = 0.02) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


__all__ = [
    "AxialBlock2D",
    "DropPath",
    "MLP",
    "PredictorAxialBlock2D",
    "SelfAttention",
    "TransformerBlock1D",
    "initialize_transformer",
]
