"""Model and optimizer construction for frame-token MotionJEPA."""
from __future__ import annotations

import logging
import torch
from model import MODEL_FACTORIES, PREDICTOR_FACTORIES, TokenLayout
from utils.schedulers import CosineWDSchedule, WarmupCosineSchedule

logger = logging.getLogger(__name__)


def position_encoding_from_model(model) -> dict:
    """Architecture metadata, not a configurable model option."""
    return {"temporal": "rope", "rope_theta": model.rotary.theta,
            "rope_time_scale": model.rotary.time_scale}


def _saved_positions(settings) -> dict:
    if not isinstance(settings, dict):
        raise ValueError("Checkpoint position_encoding must be a mapping")
    if settings.get("temporal") != "rope":
        raise ValueError("Only frame-token 1D RoPE checkpoints are supported; absolute or unspecified positions cannot be loaded")
    result = {"rope_theta": 100.0, "rope_time_scale": 1.0, **settings}
    if result != {"temporal": "rope", "rope_theta": 100.0, "rope_time_scale": 1.0}:
        raise ValueError("Checkpoint uses incompatible RoPE settings; expected theta=100 and time_scale=1")
    return result


def normalize_architecture_signature(signature: dict) -> dict:
    return {**signature, "position_encoding": _saved_positions(signature.get("position_encoding", {}))}


def _validate_config(config: dict) -> None:
    for option in ("position_encoding", "patch"):
        if option in config:
            raise ValueError(f"{option} configuration has been removed; JEPA uses frame tokens with fixed RoPE")
    meta = config.get("meta", {})
    for key, factories in (("model_name", MODEL_FACTORIES), ("predictor_name", PREDICTOR_FACTORIES)):
        if key in meta and meta[key] not in factories:
            raise ValueError(f"Unsupported {key} {meta[key]!r}; choose one of: {', '.join(factories)}")


def model_config_from_checkpoint(checkpoint: dict) -> dict:
    """Migrate compatible old RoPE metadata, never reinterpret absolute weights."""
    config = dict(checkpoint["config"])
    architecture = checkpoint.get("architecture") or {}
    candidates = [source["position_encoding"] for source in (checkpoint, architecture, config)
                  if "position_encoding" in source]
    if not candidates:
        raise ValueError("Checkpoint has no positional metadata; legacy absolute weights cannot be loaded")
    for settings in candidates:
        _saved_positions(settings)
    config.pop("position_encoding", None)
    _validate_config(config)
    for key in ("model_name", "predictor_name"):
        if key in architecture and architecture[key] != config.get("meta", {}).get(key):
            raise ValueError(f"Checkpoint architecture {key} differs from its configuration")
    return config


def init_mjepa_model(device: torch.device, num_frames: int, motion_dim: int,
                     model_name: str, predictor_name: str):
    _validate_config({"meta": {"model_name": model_name, "predictor_name": predictor_name}})
    encoder = MODEL_FACTORIES[model_name](in_chans=motion_dim, num_frames=num_frames)
    predictor = PREDICTOR_FACTORIES[predictor_name](num_frames=num_frames, embed_dim=encoder.embed_dim)
    return encoder.to(device), predictor.to(device)


def init_mjepa_model_from_config(config: dict, device: torch.device):
    _validate_config(config)
    data, meta = config["data"], config["meta"]
    return init_mjepa_model(device, int(data["num_frames"]), int(data["motion_dim"]),
                            str(meta["model_name"]),
                            str(meta["predictor_name"]))


def init_mjepa_encoder_from_config(config: dict, device: torch.device):
    _validate_config(config)
    data, meta = config["data"], config["meta"]
    return MODEL_FACTORIES[str(meta["model_name"])](
        in_chans=int(data["motion_dim"]), num_frames=int(data["num_frames"])).to(device)


def architecture_signature(encoder, predictor, *, model_name: str,
                           predictor_name: str, motion_dim: int) -> dict:
    return {"model_name": str(model_name), "predictor_name": str(predictor_name),
            "motion_dim": int(motion_dim), "encoder_layout": encoder.token_layout.signature(),
            "predictor_layout": predictor.token_layout.signature(),
            "position_encoding": position_encoding_from_model(encoder)}


def architecture_signature_from_config(config: dict) -> dict:
    _validate_config(config)
    frames = int(config["data"]["num_frames"])
    layout = TokenLayout("1d", False, frames, frames).signature()
    return {"model_name": str(config["meta"]["model_name"]),
            "predictor_name": str(config["meta"]["predictor_name"]),
            "motion_dim": int(config["data"]["motion_dim"]),
            "encoder_layout": layout, "predictor_layout": dict(layout),
            "position_encoding": {"temporal": "rope", "rope_theta": 100.0, "rope_time_scale": 1.0}}


def init_opt(
    encoder,
    predictor,
    total_steps,
    start_lr,
    ref_lr,
    warmup_steps,
    wd=1.0e-6,
    final_wd=1.0e-6,
    final_lr=0.0,
    use_float16=False,
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
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup_steps,
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
