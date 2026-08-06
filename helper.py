"""Model and optimizer construction for Motion-JEPA training."""

from __future__ import annotations

import logging

import torch

from model import (
    MODEL_FACTORIES,
    MotionTransformerPredictor1D,
    MotionTransformerPredictor2D,
)
from utils.schedulers import CosineWDSchedule, WarmupCosineSchedule


logger = logging.getLogger(__name__)


def init_mjepa_model(
    device: torch.device,
    num_frames: int,
    motion_dim: int,
    num_joints: int,
    model_name: str,
    pred_depth: int,
    pred_emb_dim: int,
):
    try:
        factory = MODEL_FACTORIES[model_name]
    except KeyError as error:
        choices = ", ".join(sorted(MODEL_FACTORIES))
        raise ValueError(f"Unknown model_name {model_name!r}; choose one of: {choices}") from error
    encoder_kwargs = {"in_chans": motion_dim, "num_frames": num_frames}
    if model_name.endswith("_2d"):
        encoder_kwargs["num_joints"] = num_joints
    encoder = factory(**encoder_kwargs)
    predictor_type = (
        MotionTransformerPredictor1D if model_name.endswith("_1d")
        else MotionTransformerPredictor2D
    )
    predictor = predictor_type(
        num_frames=num_frames,
        num_joints=num_joints,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads,
    ) if model_name.endswith("_2d") else predictor_type(
        num_frames=num_frames,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads,
    )
    return encoder.to(device), predictor.to(device)


def init_opt(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1.0e-6,
    final_wd=1.0e-6,
    final_lr=0.0,
    use_float16=False,
    ipe_scale=1.0,
):
    decay, no_decay = [], []
    for module in (encoder, predictor):
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            (no_decay if name.endswith("bias") or parameter.ndim == 1 else decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay},
            {"params": no_decay, "weight_decay": 0.0, "WD_exclude": True},
        ]
    )
    total_steps = int(ipe_scale * num_epochs * iterations_per_epoch)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=total_steps,
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True) if use_float16 else None
    logger.info("Using AdamW with %d decay and %d no-decay tensors", len(decay), len(no_decay))
    return optimizer, scaler, scheduler, wd_scheduler


__all__ = ["init_mjepa_model", "init_opt"]
