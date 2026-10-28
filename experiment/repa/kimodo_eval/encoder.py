# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from nv-tlabs/kimodo e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954.
# Batched-only evaluation adapters; see NOTICE.md for changes.
from __future__ import annotations
from typing import Optional, Tuple, Dict
import torch
from torch import nn, Tensor
from einops import repeat

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences (batch_first optional)."""

    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False) -> None:
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Note: have to replace torch.exp() and math.log() with torch.pow()
        # due to MKL exp() and ln() throws floating point exceptions on certain CPUs
        div_term = torch.pow(10000.0, -torch.arange(0, d_model, 2).float() / d_model)
        # div_term = torch.exp(
        #     torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        # )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)

class ACTORStyleEncoder(nn.Module):
    """Motion encoder in ACTOR style: optional motion_rep projection, VAE/MLP tokens, transformer."""

    def __init__(
        self,
        motion_rep: Optional[nn.Module],
        llm_shape: Optional[Tuple],
        vae: bool,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        ckpt_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.motion_rep = motion_rep
        if motion_rep is not None and llm_shape is None:
            nfeats = motion_rep.motion_rep_dim
        elif motion_rep is None and llm_shape is not None:
            nfeats = llm_shape[-1]
        else:
            raise ValueError

        self.nfeats = nfeats
        self.projection = nn.Linear(nfeats, latent_dim)

        self.vae = vae
        self.nbtokens = 2 if vae else 1
        self.tokens = nn.Parameter(torch.randn(self.nbtokens, latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout=dropout, batch_first=True)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

        self.seqTransEncoder = nn.TransformerEncoder(
            seq_trans_encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True), strict=True)

    def forward(self, x_dict: Dict) -> Tensor:
        x = x_dict["x"]
        mask = x_dict["mask"]

        x = self.projection(x)

        device = x.device
        bs = len(x)

        tokens = repeat(self.tokens, "nbtoken dim -> bs nbtoken dim", bs=bs)
        xseq = torch.cat((tokens, x), 1)

        token_mask = torch.ones((bs, self.nbtokens), dtype=bool, device=device)
        aug_mask = torch.cat((token_mask, mask), 1)

        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
        return final[:, : self.nbtokens]
