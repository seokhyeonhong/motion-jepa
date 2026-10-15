"""Frame-token 1D MotionJEPA pretraining loop."""

from __future__ import annotations

import copy
import logging
import os
import random
import shutil
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel

from dataset import make_motion_dataset
from helper import (
    architecture_signature,
    architecture_signature_from_config,
    init_mjepa_model_from_config,
    init_opt,
    normalize_architecture_signature,
    model_config_from_checkpoint,
    position_encoding_from_model,
)
from mask import MaskCollator1D
from model import MODEL_FACTORIES, PREDICTOR_FACTORIES, TokenLayout
from mask.utils import (
    apply_index_masks,
    index_mask_validity,
    repeat_mask_blocks,
)
from utils.distributed import (
    all_gather_objects,
    init_distributed,
    reduce_mean,
)
from utils.logging import AverageMeter, grad_logger
from utils.schedulers import LinearMomentumSchedule


logger = logging.getLogger(__name__)


def _unwrapped(module):
    return module.module if isinstance(module, DistributedDataParallel) else module


def _make_tensorboard_writer(log_args: dict, output: Path, purge_step: int):
    if not bool(log_args.get("tensorboard", False)):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "TensorBoard logging is enabled; install it with `pip install tensorboard`."
        ) from error
    log_dir = output / str(log_args.get("tensorboard_folder", "tensorboard"))
    return SummaryWriter(log_dir=str(log_dir), purge_step=purge_step)


def _write_tensorboard_interval(
    writer,
    *,
    global_step: int,
    loss: float,
    learning_rate: float,
    weight_decay: float,
    time_ms: float,
    memory_mib: float,
    grad_first: float,
    grad_last: float,
    grad_average: float,
) -> None:
    if writer is None:
        return
    scalars = {
        "train/loss": loss,
        "train/learning_rate": learning_rate,
        "train/weight_decay": weight_decay,
        "train/iteration_time_ms": time_ms,
        "train/gpu_memory_mib": memory_mib,
        "train/gradient_first_layer": grad_first,
        "train/gradient_last_layer": grad_last,
        "train/gradient_average": grad_average,
    }
    for name, value in scalars.items():
        writer.add_scalar(name, value, global_step)
    writer.flush()


@torch.no_grad()
def update_ema(online, target, momentum: float) -> None:
    for online_parameter, target_parameter in zip(
        _unwrapped(online).parameters(), target.parameters()
    ):
        target_parameter.mul_(momentum).add_(
            online_parameter.detach(), alpha=1.0 - momentum
        )


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _run_rank_zero(function, *, is_main: bool):
    """Propagate rank-zero failures before other ranks enter later collectives."""
    result = None
    if is_main:
        try:
            result = {"value": function(), "error": None}
        except Exception:
            result = {"value": None, "error": traceback.format_exc()}
    result = all_gather_objects(result)[0]
    if result["error"] is not None:
        raise RuntimeError(f"Rank-zero evaluation failed:\n{result['error']}")
    return result["value"]


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    """Copy an already-complete checkpoint without deserializing it."""
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _iteration_settings(args: dict) -> tuple[int, int, int, int]:
    optimization, logging_config = args["optimization"], args["logging"]
    removed = {"epochs", "warmup", "ipe_scale"}.intersection(optimization)
    if removed or "checkpoint_freq" in logging_config:
        raise ValueError("Epoch-based settings were removed; use optimization.iterations, "
                         "warmup_iterations, and logging.checkpoint_interval (all in iterations)")
    def integer(settings, key, minimum):
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
        return value
    total = integer(optimization, "iterations", 1)
    warmup = integer(optimization, "warmup_iterations", 0)
    if warmup >= total:
        raise ValueError("warmup_iterations must be smaller than iterations")
    return (total, warmup, integer(logging_config, "checkpoint_interval", 1),
            integer(logging_config, "log_freq", 1))


def _resume_signature(args: dict, batches_per_pass: int) -> dict:
    data = args["data"]
    return {
        "optimization": args["optimization"],
        "data": {key: data.get(key) for key in (
            "root_path", "meta_files", "batch_size", "num_frames", "fps",
            "motion_dim", "normalize", "stats_path", "drop_last")},
        "meta": {key: args["meta"].get(key) for key in ("seed", "use_bfloat16", "use_float16")},
        "batches_per_pass": batches_per_pass,
    }


def _save_checkpoint(
    path: Path,
    *,
    encoder,
    predictor,
    target_encoder,
    optimizer,
    scaler,
    lr_scheduler,
    wd_scheduler,
    momentum_scheduler,
    mask_collator,
    data_pass: int,
    batch_in_pass: int,
    batches_per_pass: int,
    global_step: int,
    loss: float,
    world_size: int,
    rank: int,
    config: dict,
    architecture: dict | None = None,
) -> None:
    rng_states = all_gather_objects(capture_rng_state())
    mask_states = all_gather_objects(mask_collator.state_dict())
    payload = {
        "format_version": 2,
        "encoder": _unwrapped(encoder).state_dict(),
        "predictor": _unwrapped(predictor).state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "wd_scheduler": wd_scheduler.state_dict(),
        "momentum_scheduler": momentum_scheduler.state_dict(),
        "mask_states": mask_states,
        "rng_states": rng_states,
        "data_pass": int(data_pass),
        "batch_in_pass": int(batch_in_pass),
        "resume_signature": _resume_signature(config, batches_per_pass),
        "global_step": int(global_step),
        "world_size": int(world_size),
        "loss": float(loss),
        "config": config,
        "position_encoding": position_encoding_from_model(_unwrapped(encoder)),
    }
    if architecture is not None:
        payload["architecture"] = architecture
    _run_rank_zero(lambda: _atomic_torch_save(payload, path), is_main=rank == 0)


def _load_checkpoint(
    path: Path,
    *,
    device: torch.device,
    encoder,
    predictor,
    target_encoder,
    optimizer,
    scaler,
    lr_scheduler,
    wd_scheduler,
    momentum_scheduler,
    mask_collator,
    rank: int,
    world_size: int,
    config: dict,
    batches_per_pass: int,
    architecture: dict | None = None,
) -> tuple[int, int, int]:
    # Full training checkpoints contain trusted local Python/NumPy RNG state,
    # optimizer state, and scheduler state in addition to tensor weights.
    # PyTorch 2.6 defaults weights_only=True, which cannot restore that payload.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 2:
        raise ValueError("Iteration-based resume requires a new training checkpoint; "
                         "legacy epoch checkpoints cannot be resumed with the new schedules")
    if checkpoint.get("resume_signature") != _resume_signature(config, batches_per_pass):
        raise ValueError("Resume requires the same iteration schedule, dataset, batch size, seed, and precision")
    data_pass = int(checkpoint["data_pass"])
    batch_in_pass = int(checkpoint["batch_in_pass"])
    global_step = int(checkpoint["global_step"])
    if (data_pass < 0 or not 0 <= batch_in_pass < batches_per_pass
            or data_pass * batches_per_pass + batch_in_pass != global_step
            or not 0 <= global_step <= config["optimization"]["iterations"]):
        raise ValueError("Invalid checkpoint iteration/data position")
    if int(checkpoint["world_size"]) != world_size:
        raise ValueError(
            f"Exact resume requires world_size={checkpoint['world_size']}, got {world_size}"
        )
    saved_config = model_config_from_checkpoint(checkpoint)
    if architecture is not None:
        saved_architecture = checkpoint.get("architecture")
        if saved_architecture is None:
            saved_architecture = architecture_signature_from_config(saved_config)
        if normalize_architecture_signature(saved_architecture) != normalize_architecture_signature(architecture):
            raise ValueError(
                "Checkpoint architecture differs from the requested run: "
                f"checkpoint={saved_architecture}, requested={architecture}"
            )
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    predictor.load_state_dict(checkpoint["predictor"], strict=True)
    target_encoder.load_state_dict(checkpoint["target_encoder"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None:
        if checkpoint["scaler"] is None:
            raise ValueError("Checkpoint has no scaler state for float16 resume")
        scaler.load_state_dict(checkpoint["scaler"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    wd_scheduler.load_state_dict(checkpoint["wd_scheduler"])
    momentum_scheduler.load_state_dict(checkpoint["momentum_scheduler"])
    mask_collator.load_state_dict(checkpoint["mask_states"][rank])
    restore_rng_state(checkpoint["rng_states"][rank])
    return data_pass, batch_in_pass, global_step


def _build_mask_collator(args: dict, layout: TokenLayout):
    if layout.kind != "1d" or layout.patchified:
        raise ValueError("Masking supports only 1D frame tokens")
    mask = args["mask"]
    if "version" in mask:
        raise ValueError("mask.version has been removed; omit it to use length-aware masking")
    if mask.get("strategy", "multiblock") != "multiblock":
        raise ValueError("mask.strategy must be 'multiblock'")
    return MaskCollator1D(
        num_frames=layout.token_num_frames,
        enc_frame_mask_ratio=tuple(mask["enc_frame_mask_ratio"]),
        pred_frame_mask_ratio=tuple(mask["pred_frame_mask_ratio"]),
        nenc=int(mask["num_enc_masks"]),
        npred=int(mask["num_pred_masks"]),
        allow_overlap=bool(mask["allow_overlap"]),
        context_selection=str(mask.get("context_selection", "all")),
        min_context_tokens=int(mask.get("min_context_tokens", 1)),
        min_context_ratio=float(mask.get("min_context_ratio", 0.2)),
    )


def _prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    masks_pred: list[torch.Tensor],
    *,
    num_enc_masks: int,
) -> torch.Tensor:
    """Exclude padding and weight samples equally."""
    active = index_mask_validity(masks_pred)
    if bool(active.all()):
        return F.smooth_l1_loss(prediction, target)
    active = repeat_mask_blocks(active, masks_pred[0].shape[0], num_enc_masks)
    if prediction.shape != target.shape or active.shape != prediction.shape[:2]:
        raise ValueError("Prediction, target, and target-mask shapes do not match")
    counts = active.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("Each target mask must contain at least one valid token")
    token_loss = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
    token_loss = token_loss.masked_fill(~active, 0.0)
    return (token_loss.sum(dim=1) / counts).mean()


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        resolved = torch.device(device)
    elif torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
        resolved = torch.device("cuda", local_rank)
    else:
        resolved = torch.device("cpu")
    if resolved.type == "cuda":
        torch.cuda.set_device(resolved)
    return resolved


def _seed_all(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def main(args: dict, resume_preempt: bool = False, device=None):
    total_steps, warmup_steps, checkpoint_frequency, log_frequency = _iteration_settings(args)
    if "context" in args.get("loss", {}):
        raise ValueError("loss.context has been removed; training uses masked prediction loss only")
    device = _resolve_device(device)
    distributed = init_distributed(device)
    rank, world_size = distributed.rank, distributed.world_size
    if rank != 0:
        logger.setLevel(logging.ERROR)

    seed = int(args.get("meta", {}).get("seed", 0))
    # Model and EMA-target initialization must be identical before DDP broadcasts.
    _seed_all(seed, device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    data_args = args["data"]
    meta_args = args["meta"]
    opt_args = args["optimization"]
    log_args = args["logging"]
    model_name = str(meta_args["model_name"])
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown MotionJEPA model_name: {model_name!r}")
    predictor_name = str(meta_args["predictor_name"])
    if predictor_name not in PREDICTOR_FACTORIES:
        raise ValueError(f"Unknown MotionJEPA predictor_name: {predictor_name!r}")

    output = Path(log_args["folder"])
    resume_requested = bool(meta_args.get("load_checkpoint", False) or resume_preempt)

    def initialize_output():
        # Check and create on the same rank: another rank may arrive after the
        # directory has already been created by rank zero. Broadcast failures
        # before anyone enters dataset/DDP initialization.
        if output.exists() and not resume_requested:
            raise FileExistsError(f"Output folder already exists: {output}")
        output.mkdir(parents=True, exist_ok=True)
        (output / "params-motion-jepa.yaml").write_text(
            yaml.safe_dump(args, sort_keys=False), encoding="utf-8"
        )

    encoder, predictor = init_mjepa_model_from_config(args, device)
    layout = encoder.token_layout
    mask_collator = _build_mask_collator(args, layout)
    architecture = architecture_signature(
        encoder,
        predictor,
        model_name=model_name,
        predictor_name=predictor_name,
        motion_dim=int(data_args["motion_dim"]),
    )
    _, loader, sampler = make_motion_dataset(
        root_path=data_args["root_path"],
        meta_files=data_args["meta_files"],
        batch_size=int(data_args["batch_size"]),
        num_frames=int(data_args["num_frames"]),
        fps=int(data_args["fps"]),
        motion_dim=int(data_args["motion_dim"]),
        normalize=bool(data_args.get("normalize", False)),
        stats_path=data_args.get("stats_path"),
        rank=rank,
        world_size=world_size,
        collator=None,
        drop_last=bool(data_args.get("drop_last", True)),
        num_workers=int(data_args.get("num_workers", 8)),
        pin_mem=bool(data_args.get("pin_mem", True)),
        persistent_workers=bool(data_args.get("persistent_workers", True)),
        seed=seed,
    )
    if len(loader) == 0:
        raise ValueError("Data loader has no batches; reduce batch_size or disable drop_last")

    batches_per_pass = len(loader)

    target_encoder = copy.deepcopy(encoder).to(device)
    target_encoder.requires_grad_(False)
    target_encoder.eval()

    optimizer, scaler, lr_scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        total_steps=total_steps,
        start_lr=float(opt_args["start_lr"]),
        ref_lr=float(opt_args["lr"]),
        final_lr=float(opt_args["final_lr"]),
        warmup_steps=warmup_steps,
        wd=float(opt_args["weight_decay"]),
        final_wd=float(opt_args["final_weight_decay"]),
        use_float16=bool(meta_args.get("use_float16", False)),
    )
    momentum_scheduler = LinearMomentumSchedule(
        opt_args["ema"][0], opt_args["ema"][1], total_steps
    )

    data_pass = 0
    batch_in_pass = 0
    global_step = 0
    latest_path = output / f"{log_args['write_tag']}-latest.pth.tar"
    should_load = resume_requested
    if should_load:
        read_name = meta_args.get("read_checkpoint")
        load_path = output / read_name if read_name else latest_path
        if not load_path.is_file():
            raise FileNotFoundError(f"Checkpoint requested but not found: {load_path}")
        data_pass, batch_in_pass, global_step = _load_checkpoint(
            load_path,
            device=device,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            wd_scheduler=wd_scheduler,
            momentum_scheduler=momentum_scheduler,
            mask_collator=mask_collator,
            rank=rank,
            world_size=world_size,
            architecture=architecture,
            config=args,
            batches_per_pass=batches_per_pass,
        )
        logger.info("Resumed %s at iteration=%d", load_path, global_step)

    if distributed.distributed:
        ddp_kwargs = {"broadcast_buffers": False}
        if device.type == "cuda":
            ddp_kwargs.update(device_ids=[device.index], output_device=device.index)
        encoder = DistributedDataParallel(encoder, **ddp_kwargs)
        predictor = DistributedDataParallel(predictor, **ddp_kwargs)
    if not should_load:
        # Runtime stochasticity may differ by rank after identical model creation.
        _seed_all(seed + rank, device)

    encoder_params = sum(p.numel() for p in _unwrapped(encoder).parameters())
    predictor_params = sum(p.numel() for p in _unwrapped(predictor).parameters())
    logger.info(
        "Initialized %s on %s rank=%d/%d (encoder %.2fM, predictor %.2fM)",
        model_name,
        device,
        rank,
        world_size,
        encoder_params / 1.0e6,
        predictor_params / 1.0e6,
    )

    _run_rank_zero(initialize_output, is_main=distributed.is_main)

    tensorboard_writer = None
    if distributed.is_main:
        # Purge only events after the resumed checkpoint's global step.
        tensorboard_writer = _make_tensorboard_writer(
            log_args, output, global_step + 1 if should_load else global_step
        )

    use_bfloat16 = bool(meta_args.get("use_bfloat16", False))
    use_float16 = bool(meta_args.get("use_float16", False))
    if use_bfloat16 and use_float16:
        raise ValueError("Configure only one of use_bfloat16 and use_float16")
    if use_float16 and device.type != "cuda":
        raise ValueError("Float16 training requires CUDA; use bfloat16 or float32 on CPU")
    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16 if use_float16 else None
    clip_grad = float(opt_args["clip_grad"]) if opt_args.get("clip_grad") is not None else None

    # These meters span data passes and reset only after a log event, so
    # every TensorBoard point summarizes exactly the steps since the prior one.
    interval_loss_meter = AverageMeter()
    interval_time_meter = AverageMeter()
    interval_lr_meter = AverageMeter()
    interval_wd_meter = AverageMeter()
    encoder.train()
    predictor.train()
    loss_meter = AverageMeter()
    while global_step < total_steps:
        sampler.set_epoch(data_pass)
        sampler.start_index = batch_in_pass * int(data_args["batch_size"])
        for batch in loader:
            masks_enc, masks_pred = mask_collator.sample_masks(batch[2].tolist())
            started = time.perf_counter()
            motion = batch[0].to(device=device, dtype=torch.float32, non_blocking=True)
            fps = batch[1].to(device=device, dtype=torch.float32, non_blocking=True)
            valid_length = batch[2].to(device=device, non_blocking=True)
            valid_frames = (
                torch.arange(motion.shape[1], device=device).unsqueeze(0)
                < valid_length.unsqueeze(1)
            )
            masks_enc = [mask.to(device, non_blocking=True) for mask in masks_enc]
            masks_pred = [mask.to(device, non_blocking=True) for mask in masks_pred]
            learning_rate = lr_scheduler.step()
            weight_decay = wd_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            amp_context = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if amp_dtype is not None
                else nullcontext()
            )
            with amp_context:
                with torch.no_grad():
                    target = target_encoder(motion, fps, valid_frames=valid_frames)
                    target = F.layer_norm(target, (target.shape[-1],))
                    target = apply_index_masks(target, masks_pred)
                    target = repeat_mask_blocks(target, len(motion), len(masks_enc))
                context = encoder(motion, fps, masks_enc, valid_frames=valid_frames)
                prediction = predictor(context, fps, masks_enc, masks_pred)
                loss = _prediction_loss(
                    prediction, target, masks_pred,
                    num_enc_masks=len(masks_enc),
                )

            # full precision
            if scaler is None:
                loss.backward()
                if (global_step >= warmup_steps) and (clip_grad is not None):
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
                    torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
                optimizer.step()

            # mixed precision
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if (global_step >= warmup_steps) and (clip_grad is not None):
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
                    torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()

            momentum = momentum_scheduler.step()
            update_ema(encoder, target_encoder, momentum)
            global_step += 1
            batch_in_pass += 1
            if batch_in_pass == batches_per_pass:
                data_pass += 1
                batch_in_pass = 0
            reported_loss = float(reduce_mean(loss).cpu())
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            loss_meter.update(reported_loss)
            interval_loss_meter.update(reported_loss)
            interval_time_meter.update(elapsed_ms)
            interval_lr_meter.update(learning_rate)
            interval_wd_meter.update(weight_decay)
            if distributed.is_main:
                if global_step % log_frequency == 0 or global_step == total_steps:
                    stats = grad_logger(_unwrapped(encoder).named_parameters())
                    memory = (
                        torch.cuda.max_memory_allocated(device) / 1024.0**2
                        if device.type == "cuda"
                        else 0.0
                    )
                    logger.info(
                        "iteration=%d/%d loss=%.5f lr=%.3e wd=%.3e "
                        "time=%.1fms memory=%.0fMiB grad=[%.2e, %.2e]",
                        global_step,
                        total_steps,
                        interval_loss_meter.avg,
                        interval_lr_meter.avg,
                        interval_wd_meter.avg,
                        interval_time_meter.avg,
                        memory,
                        stats.first_layer,
                        stats.last_layer,
                    )
                    _write_tensorboard_interval(
                        tensorboard_writer,
                        global_step=global_step,
                        loss=interval_loss_meter.avg,
                        learning_rate=interval_lr_meter.avg,
                        weight_decay=interval_wd_meter.avg,
                        time_ms=interval_time_meter.avg,
                        memory_mib=memory,
                        grad_first=stats.first_layer,
                        grad_last=stats.last_layer,
                        grad_average=stats.avg,
                    )
                    interval_loss_meter.reset()
                    interval_time_meter.reset()
                    interval_lr_meter.reset()
                    interval_wd_meter.reset()
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
            if not np.isfinite(reported_loss):
                raise FloatingPointError(f"Non-finite loss at step {global_step}: {reported_loss}")

            if global_step % checkpoint_frequency == 0 or global_step == total_steps:
                _save_checkpoint(
                    latest_path,
                    encoder=encoder,
                    predictor=predictor,
                    target_encoder=target_encoder,
                    optimizer=optimizer,
                    scaler=scaler,
                    lr_scheduler=lr_scheduler,
                    wd_scheduler=wd_scheduler,
                    momentum_scheduler=momentum_scheduler,
                    mask_collator=mask_collator,
                    data_pass=data_pass,
                    batch_in_pass=batch_in_pass,
                    batches_per_pass=batches_per_pass,
                    global_step=global_step,
                    loss=loss_meter.avg,
                    world_size=world_size,
                    rank=rank,
                    config=args,
                    architecture=architecture,
                )
                checkpoint_path = output / f"{log_args['write_tag']}-it{global_step}.pth.tar"
                _run_rank_zero(
                    lambda: _atomic_copy(latest_path, checkpoint_path),
                    is_main=distributed.is_main,
                )
                loss_meter.reset()
            if global_step == total_steps:
                break

    if tensorboard_writer is not None:
        tensorboard_writer.close()
    return {
        "global_step": global_step,
        "checkpoint": str(latest_path),
    }


__all__ = [
    "capture_rng_state",
    "main",
    "restore_rng_state",
    "update_ema",
]
