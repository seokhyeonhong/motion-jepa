"""Frame-token shortcut transformer, using MotionJEPA attention and RoPE."""
import math
import torch
from torch import nn
from model.modules import SelfAttention, MLP
from model.pos_embs import RotaryPosEmbed1D, temporal_token_positions
from model.specs import MODEL_SPECS


class ScalarEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, value):
        freq = torch.exp(-math.log(10000) * torch.arange(128, device=value.device).float() / 128)
        angles = value.float()[:, None] * freq[None]
        return self.mlp(torch.cat([angles.cos(), angles.sin()], -1))


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class AdaLNBlock(nn.Module):
    def __init__(self, width, heads, ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)
        self.attn = SelfAttention(width, heads)
        self.mlp = MLP(width, int(width * ratio))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))

    def forward(self, x, condition, valid, rotary):
        a, b, c, d, e, f = self.modulation(condition).chunk(6, -1)
        x = x + c[:, None] * self.attn(modulate(self.norm1(x), a, b), valid, rotary)
        x = x.masked_fill(~valid[..., None], 0)
        x = x + f[:, None] * self.mlp(modulate(self.norm2(x), d, e))
        return x.masked_fill(~valid[..., None], 0)


class MotionShortcutTransformer(nn.Module):
    def __init__(self, in_chans, num_frames=150, size="base", feature_layer=6,
                 embed_dim=None, depth=None, num_heads=None, mlp_ratio=4):
        super().__init__()
        spec = MODEL_SPECS[size]
        width = embed_dim or spec["embed_dim"]
        depth = depth or spec["depth"]
        heads = num_heads or spec["num_heads"]
        if not 1 <= feature_layer <= depth:
            raise ValueError("feature_layer must select a transformer block")
        self.in_chans, self.num_frames = in_chans, num_frames
        self.embed_dim, self.feature_layer = width, feature_layer
        self.input_proj = nn.Linear(in_chans, width)
        self.rotary = RotaryPosEmbed1D(width // heads)
        self.time_embed = ScalarEmbedding(width)
        self.step_embed = ScalarEmbedding(width)
        self.blocks = nn.ModuleList([AdaLNBlock(width, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.output_proj = nn.Linear(width, in_chans)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for embedding in (self.time_embed, self.step_embed):
            for m in embedding.mlp:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=.02)
        for m in [b.modulation[-1] for b in self.blocks] + [self.final_modulation[-1], self.output_proj]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, motion, t, step_size, fps, valid_frames, return_features=False):
        if motion.ndim != 3 or motion.shape[1:] != (self.num_frames, self.in_chans):
            raise ValueError("Motion shape does not match model")
        batch = len(motion)
        valid = valid_frames.to(device=motion.device, dtype=torch.bool)
        if valid.shape != motion.shape[:2] or not valid.any(1).all():
            raise ValueError("Every sequence must contain valid frames")
        for value, name in ((t, "time"), (step_size, "step size"), (fps, "fps")):
            if value.shape != (batch,) or not torch.isfinite(value).all():
                raise ValueError(f"Invalid {name}")
        if (t < 0).any() or (t > 1).any() or (step_size <= 0).any() or (step_size > 1).any() or (fps <= 0).any():
            raise ValueError("Invalid time, step size or FPS range")
        x = self.input_proj(motion.masked_fill(~valid[..., None], 0))
        condition = self.time_embed(t) + self.step_embed(-step_size.float().log2())
        rotary = self.rotary(temporal_token_positions(self.num_frames, fps, device=motion.device))
        features = None
        for index, block in enumerate(self.blocks, 1):
            x = block(x, condition, valid, rotary)
            if index == self.feature_layer:
                features = x
        shift, scale = self.final_modulation(condition).chunk(2, -1)
        velocity = self.output_proj(modulate(self.norm(x), shift, scale))
        velocity = velocity.masked_fill(~valid[..., None], 0)
        return (velocity, features) if return_features else velocity


class TrainingModel(nn.Module):
    """Keep the trainable projector inside the DDP forward."""
    def __init__(self, generator, teacher_dim=None):
        super().__init__()
        self.generator = generator
        self.projector = None if teacher_dim is None else nn.Sequential(
            nn.Linear(generator.embed_dim, 2048), nn.SiLU(),
            nn.Linear(2048, 2048), nn.SiLU(), nn.Linear(2048, teacher_dim))
        if self.projector is not None:
            for m in self.projector:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, motion, t, step_size, fps, valid_frames, flow_count, patch_size=1):
        velocity, features = self.generator(motion, t, step_size, fps, valid_frames, True)
        projected = None
        if self.projector is not None:
            features = features[:flow_count]
            usable = features.shape[1] // patch_size * patch_size
            features = features[:, :usable].reshape(flow_count, -1, patch_size, features.shape[-1]).mean(2)
            projected = self.projector(features)
        return velocity, projected
