"""Online JEPA attention LoRA for TMR; frozen motion caches cannot be used."""
from pathlib import Path

import torch
from torch import nn

from helper import init_mjepa_encoder_from_config
from model.motion_transformer_1d import MotionTransformer1D
from model.specs import MODEL_SPECS
from experiment.linear_probe.features import (
    _torch_load_checkpoint, _sha256_file, resolve_pretraining_stats,
)
# Reuse the tested adapter primitive; segmentation experiment files are unchanged.
from experiment.segmentation_probe.lora import DEFAULT_TARGETS, LoRALinear, inject_lora
from .model import TextMotionAlignmentV2, V2AlignmentConfig


class TextMotionAlignmentLoRA(TextMotionAlignmentV2):
    def __init__(self, config: V2AlignmentConfig, pretrained_encoder: nn.Module | None = None):
        super().__init__(config)
        if pretrained_encoder is not None:
            # Loading the original state before wrapping also verifies architecture.
            self.motion_encoder.load_state_dict(pretrained_encoder.state_dict(), strict=True)
        self.adapter_names = inject_lora(
            self.motion_encoder, rank=config.lora_rank, alpha=config.lora_alpha,
            dropout=config.lora_dropout, targets=DEFAULT_TARGETS,
        )
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.motion_encoder.eval()
        for module in self.motion_encoder.modules():
            if isinstance(module, LoRALinear):
                module.train(mode)
        return self


def load_motion_initialization(checkpoint_path, checkpoint_key, provenance, requested_size=None):
    """Strictly validate architecture, timing and the original raw normalization."""
    path = Path(checkpoint_path).expanduser().resolve()
    source = _torch_load_checkpoint(path)
    if source.get("format_version") != 1 or checkpoint_key not in source:
        raise ValueError("Unsupported JEPA checkpoint or missing encoder key")
    config = source["config"]
    name = config["meta"]["model_name"]
    sizes = {f"mot_{size}_1d": size for size in MODEL_SPECS}
    if name not in sizes:
        raise ValueError("TMR LoRA requires a non-patchified 1D RoPE JEPA encoder")
    size = sizes[name]
    if requested_size is not None and requested_size != size:
        raise ValueError("--model-size differs from the JEPA checkpoint")
    timing = provenance["provenance"]
    for key in ("num_frames", "fps", "motion_dim"):
        expected = provenance["motion_dim"] if key == "motion_dim" else timing[key]
        if config["data"][key] != expected:
            raise ValueError(f"JEPA checkpoint {key} differs from the raw dataset")
    if not config["data"].get("normalize", True):
        raise ValueError("TMR LoRA requires the pretrained normalized raw representation")
    stats = resolve_pretraining_stats(config, None)
    for filename, field in (("mean.npy", "stats_mean_sha256"), ("std.npy", "stats_std_sha256")):
        if _sha256_file(stats / filename) != provenance[field]:
            raise ValueError("Raw normalization differs from JEPA pretraining statistics")
    encoder = init_mjepa_encoder_from_config(config, torch.device("cpu"))
    if (not isinstance(encoder, MotionTransformer1D) or encoder.position_encoding != "rope"
            or encoder.rope_theta != 100.0 or encoder.rope_time_scale != 1.0):
        raise ValueError("TMR LoRA requires JEPA RoPE theta=100 and time_scale=1")
    encoder.load_state_dict(source[checkpoint_key], strict=True)
    if not all(torch.isfinite(p).all() for p in encoder.parameters()):
        raise ValueError("Pretrained JEPA weights contain non-finite values")
    epoch = source.get("next_epoch")
    if not isinstance(epoch, int) or epoch < 1:
        raise ValueError("JEPA checkpoint has no valid recorded pretraining epoch")
    record = {
        "checkpoint_path": str(path), "checkpoint_sha256": _sha256_file(path),
        "checkpoint_key": checkpoint_key, "pretrain_epoch": epoch, "model_size": size,
        "pretrain_config": config, "targets": list(DEFAULT_TARGETS),
        "normalization": "original_pretraining_raw_mean_std",
    }
    return encoder, record


def evaluation_initialization(data, saved):
    """Validate checkpoint identity without loading an unnecessary pretrained model."""
    path = Path(data["jepa_checkpoint"]).expanduser().resolve()
    if (str(path) != saved["checkpoint_path"]
            or data["checkpoint_key"] != saved["checkpoint_key"]
            or _sha256_file(path) != saved["checkpoint_sha256"]):
        raise ValueError("Evaluation JEPA initialization provenance differs from the training checkpoint")
    return dict(saved)
