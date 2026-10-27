"""Online JEPA segmentation with frozen base weights and attention LoRA adapters."""
import math

import torch
from torch import nn

from experiment.probing import FrozenFrameExtractor
from .probe import selected_feature_layers


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=None):
        super().__init__()
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError("LoRA rank must be a positive integer")
        alpha = float(rank if alpha is None else alpha)
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


class LoRASegmentation(nn.Module):
    """Entire JEPA with LoRA followed by one linear per-frame decoder."""
    def __init__(self, args, num_frames, num_classes, representation=None, device="cpu"):
        super().__init__()
        ex = FrozenFrameExtractor(args.checkpoint, None, encoder_key=args.encoder_key,
                                  device=device, representation=representation)
        selected = selected_feature_layers(args)
        final_layer = len(ex.encoder.blocks)
        if selected is not None and selected != [final_layer]:
            raise ValueError("LoRA uses only the final JEPA output; omit feature-layer arguments")
        self.encoder = ex.encoder
        self.patch = bool(ex.token_layout.patchified)
        if ex.num_frames != num_frames:
            raise ValueError("Checkpoint and segmentation frame counts differ")
        self.num_frames, self.input_dim = num_frames, ex.input_dim
        self.num_classes = num_classes
        self.patch_size = 3 if self.patch else 1
        self.register_buffer("input_mean", ex.mean.clone())
        self.register_buffer("input_std", ex.std.clone())
        rank, alpha = args.lora_rank, args.lora_alpha
        for block in self.encoder.blocks:
            block.attn.qkv = LoRALinear(block.attn.qkv, rank, alpha)
            block.attn.proj = LoRALinear(block.attn.proj, rank, alpha)
        self.head = nn.Linear(ex.feature_dim, self.patch_size * num_classes)
        self.feature_metadata = dict(ex.metadata, feature_layers=[final_layer],
                                     feature_location="final_encoder_output_after_layernorm",
                                     storage="online_no_feature_cache",
                                     normalization="pretraining_input_statistics_only")
        self.config = dict(model_version="jepa_segmentation_lora_linear_v1",
                           lora_rank=rank, lora_alpha=rank if alpha is None else alpha,
                           lora_targets=["attn.qkv", "attn.proj"],
                           adapted_blocks=list(range(1, final_layer + 1)),
                           feature_layer=final_layer, feature_dim=ex.feature_dim,
                           head="linear", output_dim=self.patch_size * num_classes,
                           num_frames=num_frames, num_classes=num_classes,
                           feature_patch_size=self.patch_size)
        self.to(device)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def extract(self, motion, valid, fps):
        if motion.ndim != 3 or motion.shape[1:] != (self.num_frames, self.input_dim):
            raise ValueError("Invalid raw motion shape")
        if valid.dtype != torch.bool or valid.shape != motion.shape[:2] or not valid.any(1).all():
            raise ValueError("Expected nonempty boolean frame masks")
        if fps.shape != (len(motion),) or not torch.isfinite(fps).all() or (fps <= 0).any():
            raise ValueError("Expected positive finite FPS")
        x = motion.masked_fill(~valid[..., None], 0)
        if not torch.isfinite(x).all():
            raise ValueError("Non-finite valid motion")
        x = ((x - self.input_mean) / self.input_std).masked_fill(~valid[..., None], 0)
        active = valid.reshape(len(x), -1, 3).any(-1) if self.patch else valid
        encoder_valid = active.repeat_interleave(3, 1) if self.patch else active
        return self.encoder(x, fps, valid_frames=encoder_valid).masked_fill(~active[..., None], 0)

    def forward(self, motion, valid, fps):
        logits = self.head(self.extract(motion, valid, fps))
        logits = logits.reshape(len(motion), self.num_frames, self.num_classes)
        return logits.masked_fill(~valid[..., None], 0)

    def parameter_counts(self):
        adapters = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        head = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        return dict(lora=adapters, head=head, total=adapters + head,
                    frozen_backbone=sum(p.numel() for p in self.encoder.parameters() if not p.requires_grad))


def prepare_lora_model(dataset, args, device, saved_model=None):
    model = LoRASegmentation(args, dataset.num_frames, dataset.num_classes,
                             getattr(dataset, "representation", None), device)
    if saved_model is not None:
        model.load_state_dict(saved_model, strict=True)
    return model
