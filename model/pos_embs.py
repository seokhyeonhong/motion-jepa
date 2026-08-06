import torch
import torch.nn as nn

class ContinuousSinCosPosEmbed1D(nn.Module):
    """
    Continuous 1D sin/cos positional embedding as an nn.Module.
    Input positions can be float tensors, e.g. [N] or [B, N].
    """
    def __init__(self, embed_dim, theta=10000.0):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {embed_dim}")
        self.embed_dim = embed_dim

        half_dim = embed_dim // 2
        omega = torch.arange(half_dim, dtype=torch.float32)
        omega = 1.0 / (theta ** (omega / float(half_dim)))  # [D/2]
        self.register_buffer("omega", omega, persistent=False)


    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        positions: float tensor with shape (...,)
        returns: tensor with shape (..., embed_dim)
        """
        pos = positions.to(dtype=self.omega.dtype)
        out = pos.unsqueeze(-1) * self.omega  # (..., D/2)
        emb = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)  # (..., D)
        return emb
