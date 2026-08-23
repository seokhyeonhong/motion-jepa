"""Model and optimizer construction for Motion-JEPA training."""

from __future__ import annotations

import logging

import torch

from model import (
    MODEL_FACTORIES,
    MODEL_KINDS,
    PATCH_MODEL_NAMES,
    PATCH_PREDICTOR_NAMES,
    PREDICTOR_FACTORIES,
    PREDICTOR_KINDS,
)
from utils.schedulers import CosineWDSchedule, WarmupCosineSchedule


logger = logging.getLogger(__name__)


def init_mjepa_model(
    device: torch.device,
    num_frames: int,
    motion_dim: int,
    num_joints: int,
    model_name: str,
    predictor_name: str,
    temporal_patch_size: int = 1,
):
    try:
        factory = MODEL_FACTORIES[model_name]
    except KeyError as error:
        choices = ", ".join(sorted(MODEL_FACTORIES))
        raise ValueError(f"Unknown model_name {model_name!r}; choose one of: {choices}") from error
    is_patch_model = model_name in PATCH_MODEL_NAMES
    encoder_kwargs = {"in_chans": motion_dim, "num_frames": num_frames}
    if is_patch_model:
        encoder_kwargs["temporal_patch_size"] = int(temporal_patch_size)
    if MODEL_KINDS[model_name] == "2d":
        encoder_kwargs["num_joints"] = num_joints
    encoder = factory(**encoder_kwargs)

    try:
        pred_factory = PREDICTOR_FACTORIES[predictor_name]
    except KeyError as error:
        choices = ", ".join(sorted(PREDICTOR_FACTORIES))
        raise ValueError(f"Unknown predictor_name {predictor_name!r}; choose one of: {choices}") from error
    is_patch_predictor = predictor_name in PATCH_PREDICTOR_NAMES
    predictor_kwargs = {"num_frames": num_frames, "embed_dim": encoder.embed_dim}
    if is_patch_predictor:
        predictor_kwargs["temporal_patch_size"] = int(temporal_patch_size)
    if PREDICTOR_KINDS[predictor_name] == "2d":
        predictor_kwargs["num_joints"] = num_joints
    predictor = pred_factory(**predictor_kwargs)
    if encoder.token_layout != predictor.token_layout:
        raise ValueError(
            "Encoder and predictor token layouts differ: "
            f"encoder={encoder.token_layout}, predictor={predictor.token_layout}"
        )
    return encoder.to(device), predictor.to(device)


def patch_size_from_config(config: dict) -> int:
    patch = config.get("patch")
    return int(patch.get("temporal_patch_size", 1)) if isinstance(patch, dict) else 1


def init_mjepa_model_from_config(config: dict, device: torch.device):
    data = config["data"]
    meta = config["meta"]
    return init_mjepa_model(
        device=device,
        num_frames=int(data["num_frames"]),
        motion_dim=int(data["motion_dim"]),
        num_joints=int(data["num_joints"]),
        model_name=str(meta["model_name"]),
        predictor_name=str(meta["predictor_name"]),
        temporal_patch_size=patch_size_from_config(config),
    )


def init_mjepa_encoder_from_config(config: dict, device: torch.device):
    data = config["data"]
    meta = config["meta"]
    model_name = str(meta["model_name"])
    try:
        factory = MODEL_FACTORIES[model_name]
    except KeyError as error:
        choices = ", ".join(sorted(MODEL_FACTORIES))
        raise ValueError(
            f"Unknown model_name {model_name!r}; choose one of: {choices}"
        ) from error
    kwargs = {
        "in_chans": int(data["motion_dim"]),
        "num_frames": int(data["num_frames"]),
    }
    if model_name in PATCH_MODEL_NAMES:
        kwargs["temporal_patch_size"] = patch_size_from_config(config)
    if MODEL_KINDS[model_name] == "2d":
        kwargs["num_joints"] = int(data["num_joints"])
    return factory(**kwargs).to(device)


def architecture_signature(
    encoder,
    predictor,
    *,
    model_name: str,
    predictor_name: str,
    motion_dim: int,
) -> dict:
    return {
        "model_name": str(model_name),
        "predictor_name": str(predictor_name),
        "motion_dim": int(motion_dim),
        "encoder_layout": encoder.token_layout.signature(),
        "predictor_layout": predictor.token_layout.signature(),
    }


def architecture_signature_from_config(config: dict) -> dict:
    data = config["data"]
    meta = config["meta"]
    model_name = str(meta["model_name"])
    predictor_name = str(meta["predictor_name"])
    patchified = model_name in PATCH_MODEL_NAMES
    predictor_patchified = predictor_name in PATCH_PREDICTOR_NAMES
    patch_size = patch_size_from_config(config) if patchified else 1
    predictor_patch_size = patch_size_from_config(config) if predictor_patchified else 1
    raw_frames = int(data["num_frames"])
    kind = MODEL_KINDS[model_name]
    predictor_kind = PREDICTOR_KINDS[predictor_name]

    def layout_signature(layout_kind: str, is_patch: bool, size: int) -> dict:
        joints = int(data["num_joints"]) if layout_kind == "2d" else None
        return {
            "kind": layout_kind,
            "patchified": is_patch,
            "raw_num_frames": raw_frames,
            "token_num_frames": raw_frames // size,
            "temporal_patch_size": size,
            "raw_num_joints": joints,
            "token_num_joints": joints,
        }

    return {
        "model_name": model_name,
        "predictor_name": predictor_name,
        "motion_dim": int(data["motion_dim"]),
        "encoder_layout": layout_signature(kind, patchified, patch_size),
        "predictor_layout": layout_signature(
            predictor_kind, predictor_patchified, predictor_patch_size
        ),
    }


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


__all__ = [
    "architecture_signature",
    "architecture_signature_from_config",
    "init_mjepa_encoder_from_config",
    "init_mjepa_model",
    "init_mjepa_model_from_config",
    "init_opt",
    "patch_size_from_config",
]
