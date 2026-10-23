"""Frozen frame features and one-layer attentive probes (no on-disk feature cache)."""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiment.utils import _sha256_file, _torch_load_checkpoint
from model import motion_transformer_1d


class FrozenFrameExtractor:
    """Apply pretraining input statistics, then return a 1-based block's output.

    All selected blocks are read before the encoder's final LayerNorm. Task
    statistics are fitted separately on these features, using training frames.
    """
    def __init__(self, checkpoint, layer, encoder_key="target_encoder", device="cpu", config_path=None, representation=None):
        if encoder_key not in ("encoder", "target_encoder"):
            raise ValueError("encoder_key must be encoder or target_encoder")
        path = Path(checkpoint).expanduser().resolve()
        saved = _torch_load_checkpoint(path)
        config = saved.get("config")
        if config_path is not None:
            import yaml
            config = yaml.safe_load(Path(config_path).read_text())
        if not isinstance(config, dict):
            raise ValueError("Checkpoint must contain its pretraining config")
        data = config["data"]
        if representation is not None:
            from motion_rep.registry import get_representation
            expected = get_representation(representation)
            actual = get_representation(data.get("representation", "MotionJEPAMotionRep"))
            if actual is not expected or int(data["motion_dim"]) != expected.FEATURE_DIM:
                raise ValueError("Dataset representation differs from the JEPA checkpoint representation")
        name = config["meta"]["model_name"]
        factory = getattr(motion_transformer_1d, name, None)
        if factory is None or not name.startswith("mot_") or "predictor" in name:
            raise ValueError(f"Unsupported encoder factory: {name}")
        self.encoder = factory(in_chans=int(data["motion_dim"]), num_frames=int(data["num_frames"]))
        if isinstance(layer, bool) or not isinstance(layer, int) or not 1 <= layer <= len(self.encoder.blocks):
            raise ValueError("feature_layer must be a 1-based transformer block index")
        state = saved[encoder_key]
        state = {key.removeprefix("module."): value for key, value in state.items()}
        self.encoder.load_state_dict(state, strict=True)
        self.device = torch.device(device)
        self.feature_dim = self.encoder.embed_dim
        self.num_frames = self.encoder.num_frames
        self.input_dim = self.encoder.in_chans
        self.layer = layer
        self.mean = torch.zeros(self.input_dim, device=self.device)
        self.std = torch.ones(self.input_dim, device=self.device)
        stats = None
        if data.get("normalize", False):
            root = Path(data["root_path"]).expanduser().resolve()
            stats = Path(data.get("stats_path") or "stats")
            if not stats.is_absolute():
                stats = root / stats
            if stats.suffix:
                stats = stats.parent
            mean = np.load(stats / "mean.npy", allow_pickle=False).astype(np.float32)
            std = np.load(stats / "std.npy", allow_pickle=False).astype(np.float32)
            if mean.shape != (self.input_dim,) or std.shape != mean.shape:
                raise ValueError("Pretraining input statistics have the wrong width")
            if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std < 0).any():
                raise ValueError("Pretraining input statistics must be finite with nonnegative std")
            self.mean = torch.as_tensor(mean, device=self.device)
            self.std = torch.as_tensor(np.where(std < 1e-6, 1., std), device=self.device)
        self.metadata = {
            "checkpoint": str(path), "checkpoint_sha256": _sha256_file(path),
            "checkpoint_step": saved.get("global_step"), "encoder_key": encoder_key,
            "feature_layer": layer, "feature_dim": self.feature_dim,
            "feature_location": "transformer_block_output_before_final_layernorm",
            "pretraining_input_stats": str(stats) if stats else None,
            "extraction_dtype": "float32", "cache_storage": "anonymous_cpu_ram_only",
        }
        if stats is not None:
            self.metadata["pretraining_stats_sha256"] = {
                k: _sha256_file(stats / (k + ".npy")) for k in ("mean", "std")
            }
        del state, saved
        self.encoder.requires_grad_(False).eval().to(self.device)

    @torch.inference_mode()
    def __call__(self, motion, fps, lengths):
        if self.encoder is None:
            raise RuntimeError("Frozen extractor was released after preprocessing")
        motion = torch.as_tensor(motion, device=self.device, dtype=torch.float32)
        lengths = torch.as_tensor(lengths, device=self.device)
        if motion.ndim != 3 or motion.shape[1:] != (self.num_frames, self.input_dim):
            raise ValueError("Raw motion shape differs from the checkpoint")
        if lengths.shape != (len(motion),) or ((lengths < 1) | (lengths > self.num_frames)).any():
            raise ValueError("Invalid per-clip frame lengths")
        active = torch.arange(self.num_frames, device=self.device)[None] < lengths[:, None]
        motion = motion.masked_fill(~active[..., None], 0)
        if not torch.isfinite(motion).all():
            raise ValueError("Valid raw motion frames must be finite")
        motion = ((motion - self.mean) / self.std).masked_fill(~active[..., None], 0)
        fps = torch.as_tensor(fps, device=self.device, dtype=torch.float32)
        if fps.shape != (len(motion),) or not torch.isfinite(fps).all() or (fps <= 0).any():
            raise ValueError("Expected a positive finite FPS for every clip")
        return self.encoder(motion, fps, valid_frames=active, feature_layer=self.layer).float()

    def release(self):
        """Drop all encoder/device references before constructing trainable probes."""
        self.encoder = None
        self.mean = None
        self.std = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


class FeatureMoments:
    """Stable population mean/std over valid training frames, in CPU float64."""
    def __init__(self, feature_dim):
        self.count = 0
        self.mean = torch.zeros(feature_dim, dtype=torch.float64)
        self.m2 = torch.zeros_like(self.mean)

    def update(self, features, valid):
        features = torch.as_tensor(features).detach()
        valid = torch.as_tensor(valid, device=features.device, dtype=torch.bool)
        if features.ndim != 3 or valid.shape != features.shape[:2] or features.shape[-1] != len(self.mean):
            raise ValueError("Feature statistics expect matching [B,T,D] features and [B,T] validity")
        x = features[valid].to(device="cpu", dtype=torch.float64)
        if not len(x):
            return
        if not torch.isfinite(x).all():
            raise ValueError("Training feature statistics contain nonfinite values")
        n = len(x)
        batch_mean = x.mean(0)
        batch_m2 = ((x - batch_mean) ** 2).sum(0)
        delta = batch_mean - self.mean
        total = self.count + n
        self.m2 += batch_m2 + delta.square() * (self.count * n / total)
        self.mean += delta * (n / total)
        self.count = total

    def finalize(self):
        if not self.count:
            raise ValueError("Cannot normalize features without valid training frames")
        std = (self.m2 / self.count).clamp_min(0).sqrt()
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        return self.mean.float(), std.float()


class AttentiveProbe(nn.Module):
    """One attention operation and one task projection, with fixed normalization.

    Segmentation uses each frame as query plus a residual to retain frame identity.
    Retrieval uses one learned query to pool the sequence. There is no feedforward
    transformer block, extra MLP, or trainable backbone.
    """
    def __init__(self, input_dim, output_dim, mean, std, num_heads=6, pool=False):
        super().__init__()
        if input_dim < 1 or output_dim < 1 or num_heads < 1 or input_dim % num_heads:
            raise ValueError("Positive dimensions required and heads must divide input_dim")
        mean, std = torch.as_tensor(mean).float(), torch.as_tensor(std).float()
        if mean.shape != (input_dim,) or std.shape != mean.shape:
            raise ValueError("Feature normalization must match input_dim")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Feature normalization must be finite with positive std")
        self.input_dim, self.output_dim, self.pool = input_dim, output_dim, pool
        self.register_buffer("feature_mean", mean.clone())
        self.register_buffer("feature_std", std.clone())
        self.attention = nn.MultiheadAttention(input_dim, num_heads, dropout=0., batch_first=True)
        if pool:
            self.query = nn.Parameter(torch.empty(1, 1, input_dim))
            nn.init.normal_(self.query, std=0.02)
        self.projection = nn.Linear(input_dim, output_dim)
        nn.init.trunc_normal_(self.projection.weight, std=0.02)
        nn.init.zeros_(self.projection.bias)

    def forward(self, features, valid):
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError("Invalid attentive probe feature shape")
        valid = torch.as_tensor(valid, device=features.device, dtype=torch.bool)
        if valid.shape != features.shape[:2] or not valid.any(1).all():
            raise ValueError("Every probe sequence must contain at least one valid frame")
        x = features.float().masked_fill(~valid[..., None], 0)
        x = ((x - self.feature_mean) / self.feature_std).masked_fill(~valid[..., None], 0)
        query = self.query.expand(len(x), -1, -1) if self.pool else x
        attended, _ = self.attention(query, x, x, key_padding_mask=~valid, need_weights=False)
        if self.pool:
            return self.projection(attended[:, 0])
        return self.projection(x + attended).masked_fill(~valid[..., None], 0)
