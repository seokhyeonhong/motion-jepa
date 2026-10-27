"""Online JEPA attention LoRA with an independent motion retrieval Transformer."""
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

from experiment.probing import FrozenFrameExtractor
from experiment.segmentation.lora import LoRALinear
from .model import SequenceEncoder


class LoRAMotionEncoder(nn.Module):
    def __init__(self, config, options, representation=None, device="cpu"):
        super().__init__()
        ex = FrozenFrameExtractor(options["checkpoint"], None,
                                  encoder_key=options["encoder_key"],
                                  representation=representation, device=device)
        if ex.num_frames != config.num_frames or ex.input_dim != config.motion_dim:
            raise ValueError("JEPA checkpoint and raw TMR dimensions differ")
        self.encoder = ex.encoder
        self.num_frames, self.input_dim, self.fps = ex.num_frames, ex.input_dim, config.fps
        self.register_buffer("input_mean", ex.mean.clone())
        self.register_buffer("input_std", ex.std.clone())
        for block in self.encoder.blocks:
            block.attn.qkv = LoRALinear(block.attn.qkv, options["rank"], options["alpha"])
            block.attn.proj = LoRALinear(block.attn.proj, options["rank"], options["alpha"])
        patch_size = ex.token_layout.temporal_patch_size
        self.head = SequenceEncoder(ex.feature_dim, config,
                                    num_tokens=config.num_frames // patch_size,
                                    fps=config.fps / patch_size, depth=config.motion_depth)
        self.metadata = dict(ex.metadata,
                             feature_location="final_encoder_output_after_layernorm",
                             cache_storage="online_no_feature_cache",
                             extraction_dtype="training_autocast_setting",
                             normalization="pretraining_input_statistics_only",
                             lora_targets=["attn.qkv", "attn.proj"],
                             adapted_blocks=list(range(1, len(self.encoder.blocks) + 1)),
                             rank=options["rank"], alpha=options["alpha"])
        self.to(device)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, tokens, valid):
        if tokens.ndim != 3 or tokens.shape[-1] != self.input_dim or not 0 < tokens.shape[1] <= self.num_frames:
            raise ValueError("Invalid raw TMR motion dimensions")
        if valid.dtype != torch.bool or valid.shape != tokens.shape[:2]:
            raise ValueError("Expected boolean motion mask")
        padding = self.num_frames - tokens.shape[1]
        active = F.pad(valid, (0, padding), value=False)
        x = F.pad(tokens.masked_fill(~valid[..., None], 0), (0, 0, 0, padding))
        if not torch.isfinite(x).all():
            raise ValueError("Non-finite valid raw motion")
        x = ((x - self.input_mean) / self.input_std).masked_fill(~active[..., None], 0)
        token_active = self.encoder.token_layout.valid_token_mask(active)
        if not token_active.any(1).all():
            raise ValueError("Each sequence needs a complete motion patch")
        fps = torch.full((len(x),), self.fps, device=x.device, dtype=torch.float32)
        features = self.encoder(x.to(self.encoder.input_proj.weight.dtype), fps, valid_frames=active)
        return self.head(features, token_active)


def lora_configuration(args):
    return dict(checkpoint=str(Path(args.checkpoint).expanduser().resolve()),
                encoder_key=args.encoder_key, rank=args.lora_rank,
                alpha=args.lora_rank if args.lora_alpha is None else args.lora_alpha)


def attach_lora(model, datasets, provenance, options, device):
    for dataset in datasets.values():
        if any(float(record["fps"]) != model.config.fps for record in dataset.records):
            raise ValueError("TMR records differ from the configured motion FPS")
        dataset.raw_dataset.normalize = False
    model.motion_encoder = LoRAMotionEncoder(
        model.config, options, provenance["provenance"].get("representation"), device)
    return dict(provenance, lora=model.motion_encoder.metadata)
