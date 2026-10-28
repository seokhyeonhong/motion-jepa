"""Train text-motion retrieval using raw motion and cached text tokens."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from experiment.utils import (
    PROJECT_ROOT,
    _atomic_json_save,
    _atomic_torch_save,
    _seed_all,
    _torch_load_checkpoint,
    resolve_device,
)
from utils.schedulers import WarmupCosineSchedule
from .dataset import (
    DEFAULT_ANNOTATIONS_PATH,
    DEFAULT_CACHE_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_TEXT_MODEL,
    collate_pairs,
)
from .features import load_prepared_datasets
from .losses import symmetric_multi_positive_info_nce
from .model import (
    AlignmentConfig, TextMotionAlignment, MODEL_SPECS, parameter_counts,
)
from .retrieval import evaluate_retrieval


DATA_FIELDS = (
    "dataset_root", "annotations_path", "cache_root",
    "stats_path", "text_model",
    "text_revision", "max_text_length", "text_adapter", "text_adapter_revision",
)
OPTIMIZATION_FIELDS = (
    "seed", "iterations", "warmup_iterations", "eval_interval", "checkpoint_interval",
    "log_freq", "batch_size", "start_lr", "lr", "final_lr",
    "weight_decay", "gradient_clip", "temperature", "use_bfloat16",
)


def shuffled_training_batches(size, batch_size, generator):
    """Repeat shuffled passes, dropping each pass's incomplete final batch."""
    if size < batch_size or batch_size < 1:
        raise ValueError("Training requires at least one complete batch")
    while True:
        order = torch.randperm(size, generator=generator).tolist()
        for start in range(0, size - batch_size + 1, batch_size):
            yield order[start:start + batch_size]


def optimization_configuration(args):
    values = {name: getattr(args, name) for name in OPTIMIZATION_FIELDS}
    if getattr(args, "validation_only", False):
        values["validation_only"] = True
    return values


def data_configuration(args: argparse.Namespace) -> dict[str, Any]:
    from experiment.representation import select_defaults
    select_defaults(args, "tmr")
    values = {name: getattr(args, name, None) for name in DATA_FIELDS}
    for name in ("dataset_root", "annotations_path", "cache_root", "stats_path"):
        if values[name] is not None:
            values[name] = str(Path(values[name]).expanduser().resolve())
    values["text_model"] = str(values["text_model"])
    if getattr(args, "representation", None) is not None:
        values["representation"] = args.representation
    if args.train_fraction != 1:
        values.update(train_fraction=args.train_fraction, train_subset_seed=args.seed)
    return values


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.inference_mode()
def evaluate_split(
    model: TextMotionAlignment,
    dataset,
    device: torch.device,
    *,
    batch_size: int = 256,
    num_workers: int = 0,
    use_bfloat16: bool = True,
    chunk_size: int = 512,
) -> dict[str, float]:
    """Encode all motions and each distinct caption once, then rank the full gallery."""
    if not len(dataset):
        raise ValueError("Retrieval evaluation requires a nonempty split")
    dataset = copy.copy(dataset)
    dataset.sample_captions = False
    model.eval()
    if getattr(dataset, "motion_bank", None) is not None:
        num_workers = 0
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_pairs, pin_memory=device.type == "cuda",
    )
    motion_parts, text_parts = [], []
    motion_ids = dataset.caption_candidates
    text_ids = list(dict.fromkeys(caption for candidates in motion_ids for caption in candidates))
    for batch in loader:
        with _autocast(device, use_bfloat16):
            motion = model.encode_motion(
                batch["motion_tokens"].to(device, non_blocking=True),
                batch["motion_mask"].to(device, non_blocking=True),
            )
        motion_parts.append(motion.float().cpu())
    for start in range(0, len(text_ids), batch_size):
        tokens = [dataset.text_bank[caption_id] for caption_id in text_ids[start:start + batch_size]]
        padded = pad_sequence(tokens, batch_first=True).to(device)
        lengths = torch.tensor([len(value) for value in tokens], device=device)
        mask = torch.arange(padded.shape[1], device=device)[None, :] < lengths[:, None]
        with _autocast(device, use_bfloat16):
            text = model.encode_text(padded, mask)
        text_parts.append(text.float().cpu())
    return evaluate_retrieval(
        torch.cat(motion_parts).to(device), torch.cat(text_parts).to(device),
        motion_ids, text_ids, chunk_size=chunk_size,
    )


def _capture_rng(loader_generator: torch.Generator, device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        "loader_generator": loader_generator.get_state(),
    }


def _restore_rng(state: dict[str, Any], loader_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    loader_generator.set_state(state["loader_generator"])


def _write_metrics(history: list[dict[str, Any]], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        if history:
            writer = csv.DictWriter(file, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    os.replace(temporary, path)


def _tensorboard_writer(args, output: Path, global_step: int):
    if not args.tensorboard:
        return nullcontext(None)
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise RuntimeError(
            "TensorBoard logging requires tensorboard; install it in motion-jepa with "
            "python -m pip install -r experiment/tmr/requirements.txt, or use --no-tensorboard"
        ) from error
    directory = Path(args.tensorboard_dir or output / "tensorboard").expanduser().resolve()
    # The saved checkpoint includes evaluation at global_step. Discard only later
    # events from uncommitted optimizer steps when resuming an interrupted run.
    return SummaryWriter(log_dir=str(directory), purge_step=global_step + 1 if args.resume else None)


def load_alignment_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = _torch_load_checkpoint(path)
    if checkpoint.get("experiment") != "tmr_alignment":
        raise ValueError(f"Not a supported TMR alignment checkpoint: {path}")
    for key in ("model", "config", "provenance"):
        if key not in checkpoint:
            raise ValueError(f"TMR checkpoint has no {key!r}: {path}")
    return checkpoint


def _validate_args(args: argparse.Namespace) -> None:
    if args.probe_architecture == "mlp" and args.mode != "probe":
        raise ValueError("--probe-architecture mlp requires --mode probe")
    if args.mode == "lora":
        if args.checkpoint is None or args.feature_source != "jepa":
            raise ValueError("LoRA requires a JEPA checkpoint")
        if args.feature_layer is not None or args.feature_layers is not None or args.patchify:
            raise ValueError("LoRA uses the final JEPA output and checkpoint tokenization")
        if args.lora_rank < 1 or (args.lora_alpha is not None and
                (not math.isfinite(args.lora_alpha) or args.lora_alpha <= 0)):
            raise ValueError("LoRA rank and alpha must be positive and finite")
    if getattr(args, "motion_depth", None) is not None:
        if args.motion_depth < 1:
            raise ValueError("--motion-depth must be positive")
        if args.mode == "probe" and args.probe_architecture not in {"transformer", "mlp"}:
            raise ValueError("--motion-depth requires --probe-architecture transformer or mlp in probe mode")
    if getattr(args, "feature_layers", None) is not None and (args.mode != "probe" or args.feature_source != "jepa"):
        raise ValueError("--feature-layers requires JEPA probe mode")
    if getattr(args, "mode", "supervised") == "probe":
        if getattr(args, "patchify", False):
            raise ValueError("--patchify is only supported in supervised raw-motion mode")
        if args.precompute_batch_size < 1 or args.probe_heads < 1:
            raise ValueError("Precompute batch size and probe head count must be positive")
        if args.feature_source == "jepa":
            from .probing import selected_feature_layers
            selected_feature_layers(args)
            if args.checkpoint is None:
                raise ValueError("JEPA probing requires --checkpoint")
        if args.feature_source == "raw" and args.checkpoint is not None:
            raise ValueError("Raw probing does not use --checkpoint")
    if not math.isfinite(args.train_fraction) or not 0 < args.train_fraction <= 1:
        raise ValueError("--train-fraction must be finite and in (0, 1]")
    if args.iterations <= 0 or not 0 <= args.warmup_iterations < args.iterations:
        raise ValueError("Require iterations > 0 and 0 <= warmup_iterations < iterations")
    if min(args.eval_interval, args.checkpoint_interval, args.log_freq) <= 0:
        raise ValueError("Evaluation, checkpoint, and logging intervals must be positive")
    for name in ("start_lr", "lr", "final_lr"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if args.lr == 0 or max(args.start_lr, args.final_lr) > args.lr:
        raise ValueError("Require positive lr with start_lr and final_lr no greater than lr")
    if args.batch_size < 2 or args.eval_batch_size < 1 or args.num_workers < 0 or args.chunk_size < 1:
        raise ValueError("Training batch size must be >=2; evaluation sizes positive; workers >=0")
    for name in ("temperature",):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("weight_decay", "gradient_clip"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    probe_mode = getattr(args, "mode", "supervised") == "probe"
    default_output = PROJECT_ROOT / "output/tmr" / args.model_size
    if getattr(args, "representation", None) == "LocalMotionJEPARep":
        default_output /= "local-motion-jepa"
    if probe_mode:
        from .probing import selected_feature_layers
        layers = selected_feature_layers(args) if args.feature_source == "jepa" else None
        feature_name = ("layer-" if len(layers) == 1 else "layers-") + "-".join(map(str, layers)) if layers else "raw"
        default_output /= f"probe-{feature_name}"
        if getattr(args, "probe_architecture", "attentive") == "transformer":
            default_output /= "transformer"
    if args.mode == "lora":
        default_output /= f"lora-rank-{args.lora_rank}"
    if getattr(args, "patchify", False):
        default_output /= "patchify"
    if args.train_fraction != 1:
        default_output /= f"frac{args.train_fraction:g}"
    if getattr(args, "motion_depth", None) is not None:
        default_output /= f"motion-depth-{args.motion_depth}"
    default_output /= f"seed-{args.seed}"
    output = Path(args.output_root or default_output)
    output = output.expanduser().resolve()
    latest_path, best_path = output / "latest.pth.tar", output / "best.pth.tar"
    existing = [name for name in ("config.json", "metrics.csv", "latest.pth.tar", "best.pth.tar", "summary.json") if (output / name).exists()]
    if existing and not args.resume:
        raise FileExistsError(f"TMR output already exists under {output}; use --resume or another --output-root")
    if args.resume and not latest_path.is_file():
        raise FileNotFoundError(f"--resume requires {latest_path}")
    device = resolve_device(args.device)
    _seed_all(args.seed)
    data_config = data_configuration(args)
    datasets, provenance = load_prepared_datasets(**data_config)
    if len(datasets["train"]) < 2 or not len(datasets["val"]):
        raise ValueError("TMR needs at least two training pairs and a nonempty validation split")
    if len(datasets["train"]) < args.batch_size:
        raise ValueError("Training subset is smaller than batch_size; choose a common batch size for all fractions")
    probe_config, motion_mean, motion_std = None, None, None
    if probe_mode:
        from .probing import prepare_probe_datasets, probe_configuration
        probe_config = probe_configuration(args)
        datasets, provenance, motion_mean, motion_std = prepare_probe_datasets(
            datasets, provenance, probe_config, device,
            batch_size=args.precompute_batch_size, num_workers=args.num_workers,
        )
        # Avoid replicating the large anonymous RAM bank in training workers.
        args.num_workers = 0
        _seed_all(args.seed)
    transformer_probe = probe_mode and getattr(args, "probe_architecture", "attentive") == "transformer"
    mlp_probe = probe_mode and args.probe_architecture == "mlp"
    layout = provenance.get("probe", {}).get("extractor", {}).get("token_layout", {})
    patch_size = layout.get("temporal_patch_size", 1) if (transformer_probe or mlp_probe) else 1
    model_config = AlignmentConfig(
        text_dim=provenance["text_dim"], motion_dim=provenance["motion_dim"],
        model_size=args.model_size,
        num_frames=provenance["provenance"]["num_frames"] // patch_size,
        fps=provenance["provenance"]["fps"] / patch_size,
        motion_encoder="mlp_probe" if mlp_probe else "transformer_probe" if transformer_probe else "attentive_probe" if probe_mode else "transformer",
        probe_heads=args.probe_heads if probe_mode else 6,
        patchify=getattr(args, "patchify", False),
        motion_depth=getattr(args, "motion_depth", None),
    )
    model_settings = asdict(model_config)
    if model_config.motion_depth is None:
        model_settings.pop("motion_depth")
    if not model_config.patchify:
        model_settings.pop("patchify")
    if not probe_mode:
        # Preserve the existing supervised checkpoint configuration exactly.
        model_settings.pop("motion_encoder")
        model_settings.pop("probe_heads")
    config = {
        "model": model_settings, "data": data_config,
        "optimization": optimization_configuration(args),
    }
    if probe_config is not None:
        config["probe"] = probe_config
    model = TextMotionAlignment(model_config, motion_mean=motion_mean, motion_std=motion_std).to(device)
    if args.mode == "lora":
        from .lora import attach_lora, lora_configuration
        config["lora"] = lora_configuration(args)
        provenance = attach_lora(model, datasets, provenance, config["lora"], device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineSchedule(
        optimizer, warmup_steps=args.warmup_iterations, start_lr=args.start_lr,
        ref_lr=args.lr, final_lr=args.final_lr, T_max=args.iterations,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        datasets["train"],
        batch_sampler=shuffled_training_batches(len(datasets["train"]), args.batch_size, generator),
        num_workers=args.num_workers, generator=generator, collate_fn=collate_pairs,
        persistent_workers=args.num_workers > 0, pin_memory=device.type == "cuda",
    )
    best_step, best_score, global_step = 0, -math.inf, 0
    weighted_loss, sample_count = 0.0, 0
    history: list[dict[str, Any]] = []
    if args.resume:
        checkpoint = load_alignment_checkpoint(latest_path)
        if (checkpoint["config"] != config
                or checkpoint["provenance"] != provenance):
            raise ValueError("Resume config or feature provenance differs from the saved run")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best_step, best_score = checkpoint["best_step"], checkpoint["best_val_mean_r1"]
        global_step = checkpoint["global_step"]
        weighted_loss, sample_count = checkpoint["train_accumulator"]
        history = checkpoint["history"]
        _restore_rng(checkpoint["rng_state"], generator)
        if best_step and not best_path.is_file():
            raise FileNotFoundError(f"Resume requires the selected checkpoint: {best_path}")
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json_save(config, output / "config.json")
    _atomic_json_save(provenance, output / "provenance.json")
    if "train_subset" in provenance:
        _atomic_json_save({
            **provenance["train_subset"], "sample_ids": datasets["train"].sample_ids,
        }, output / "train-subset.json")
    print(
        f"TMR pairs: train={len(datasets['train'])} (fraction={args.train_fraction:g}), "
        f"val={len(datasets['val'])}, test={len(datasets['test'])}", flush=True,
    )
    counts = parameter_counts(model)
    print(f"TMR architecture: {asdict(model_config)}", flush=True)
    print(f"TMR trainable parameters: {counts}", flush=True)
    _atomic_json_save(counts, output / "parameters.json")
    _atomic_json_save({
        "python": sys.executable, "prefix": sys.prefix, "torch": str(torch.__version__),
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }, output / "runtime.json")
    _write_metrics(history, output / "metrics.csv")
    with _tensorboard_writer(args, output, global_step) as writer:
        started = time.monotonic()
        batches = iter(loader)
        for global_step in range(global_step + 1, args.iterations + 1):
            batch = next(batches)
            model.train()
            learning_rate = scheduler.step()
            count = len(batch["caption_ids"])
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, args.use_bfloat16):
                motion, text = model(
                    batch["motion_tokens"].to(device, non_blocking=True),
                    batch["motion_mask"].to(device, non_blocking=True),
                    batch["text_tokens"].to(device, non_blocking=True),
                    batch["text_mask"].to(device, non_blocking=True),
                )
                loss = symmetric_multi_positive_info_nce(
                    motion, text, batch["caption_ids"], batch["source_ids"],
                    batch["start_frames"], batch["end_frames"], temperature=args.temperature,
                    caption_candidate_ids=batch["caption_candidate_ids"],
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite TMR loss at iteration {global_step}")
            loss.backward()
            if args.gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            loss_value = float(loss.detach())
            weighted_loss += loss_value * count
            sample_count += count
            if writer is not None:
                writer.add_scalar("train/loss", loss_value, global_step)
                writer.add_scalar("train/learning_rate", learning_rate, global_step)
            if global_step % args.log_freq == 0 or global_step == args.iterations:
                print(f"TMR iteration={global_step}/{args.iterations} "
                      f"loss={weighted_loss / sample_count:.5f} lr={learning_rate:.3e}", flush=True)
            improved = False
            if global_step % args.eval_interval == 0 or global_step == args.iterations:
                validation = evaluate_split(
                    model, datasets["val"], device, batch_size=args.eval_batch_size,
                    num_workers=args.num_workers, use_bfloat16=args.use_bfloat16, chunk_size=args.chunk_size,
                )
                row = {
                    "iteration": global_step, "lr": learning_rate,
                    "train_loss": weighted_loss / sample_count, "train_samples": sample_count,
                    **{f"val_{key}": value for key, value in validation.items()},
                }
                history.append(row)
                improved = validation["mean_r1"] > best_score
                if improved:
                    best_step, best_score = global_step, float(validation["mean_r1"])
                if writer is not None:
                    writer.add_scalar("train/interval_loss", row["train_loss"], global_step)
                    writer.add_scalar("train/samples_seen", global_step * args.batch_size, global_step)
                    for key, value in validation.items():
                        writer.add_scalar(f"val/{key}", value, global_step)
                    writer.add_scalar("val/best_mean_r1", best_score, global_step)
                    writer.add_scalar("val/best_step", best_step, global_step)
                weighted_loss, sample_count = 0.0, 0
                _write_metrics(history, output / "metrics.csv")
                print(f"TMR validation iteration={global_step} mean_R@1={validation['mean_r1']:.4f} "
                      f"best={best_score:.4f} elapsed={time.monotonic() - started:.1f}s", flush=True)
            if improved or global_step % args.checkpoint_interval == 0 or global_step == args.iterations:
                checkpoint = {
                    "experiment": "tmr_alignment", "config": config,
                    "provenance": provenance, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "global_step": global_step,
                    "best_step": best_step, "best_val_mean_r1": best_score,
                    "train_accumulator": (weighted_loss, sample_count),
                    "history": history, "rng_state": _capture_rng(generator, device),
                }
                if improved:
                    _atomic_torch_save(checkpoint, best_path)
                _atomic_torch_save(checkpoint, latest_path)
                if writer is not None:
                    writer.flush()
        selected = load_alignment_checkpoint(best_path)
        if (selected["config"] != config
                or selected["provenance"] != provenance):
            raise ValueError("Best checkpoint does not belong to this TMR run")
        model.load_state_dict(selected["model"], strict=True)
        test = evaluate_split(
            model, datasets["test"], device, batch_size=args.eval_batch_size,
            num_workers=args.num_workers, use_bfloat16=args.use_bfloat16, chunk_size=args.chunk_size,
        ) if len(datasets["test"]) and not getattr(args, "validation_only", False) else None
        if writer is not None and test is not None:
            for key, value in test.items():
                writer.add_scalar(f"test/{key}", value, global_step)
        summary = {
            "input_source": args.feature_source if probe_mode else "raw", "output_root": str(output),
            "train_fraction": args.train_fraction,
            "split_counts": {split: len(dataset) for split, dataset in datasets.items()},
            "selection": "mean_bidirectional_val_r1", "best_step": best_step,
            "best_val": {key.removeprefix("val_"): value for key, value in
                         next(row for row in history if row["iteration"] == best_step).items() if key.startswith("val_")},
            "test": test, "iterations_completed": global_step, "global_step": global_step,
            "samples_seen": global_step * args.batch_size,
            "equivalent_epochs": global_step * args.batch_size / len(datasets["train"]),
            "trainable_parameters": counts["total"],
            "parameter_counts": counts, "model": model_settings,
            **({"probe": probe_config} if probe_mode else {}),
            **({"lora": config["lora"]} if args.mode == "lora" else {}),
            "best_checkpoint": str(best_path), "latest_checkpoint": str(latest_path),
            "tensorboard_dir": str(Path(args.tensorboard_dir or output / "tensorboard").expanduser().resolve()) if args.tensorboard else None,
        }
        _atomic_json_save(summary, output / "summary.json")
        return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    from experiment.representation import add_argument
    add_argument(parser)
    parser.add_argument("--patchify", action=argparse.BooleanOptionalAction, default=False,
                        help="Use non-overlapping three-frame motion patches in supervised mode; text is unchanged.")
    parser.add_argument("--mode", choices=("supervised", "probe", "lora"), default="supervised")
    parser.add_argument("--feature-source", choices=("raw", "jepa"), default="jepa")
    layer_options = parser.add_mutually_exclusive_group()
    layer_options.add_argument("--feature-layer", type=int, help="One-based frozen JEPA block output; defaults to 12.")
    layer_options.add_argument("--feature-layers", type=int, nargs="+", help="Ordered JEPA block outputs to concatenate along channels.")
    parser.add_argument("--checkpoint", type=Path, help="JEPA checkpoint, required for frozen probing or online LoRA.")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, help="Defaults to rank (scale 1).")
    parser.add_argument("--encoder-key", default="target_encoder", choices=("encoder", "target_encoder"))
    parser.add_argument("--precompute-batch-size", type=int, default=256)
    parser.add_argument("--probe-architecture", choices=("attentive", "transformer", "mlp"), default="attentive")
    parser.add_argument("--probe-heads", type=int, default=6)
    parser.add_argument("--model-size", choices=tuple(MODEL_SPECS), default="base", help="Text and motion encoder size; defaults to base.")
    parser.add_argument("--motion-depth", type=int, help="Override motion Transformer blocks or MLP hidden Linear+GELU stages (MLP default: 2); text encoder retains the size preset.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--annotations-path", type=Path, default=DEFAULT_ANNOTATIONS_PATH)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--stats-path", type=Path)
    parser.add_argument(
        "--text-model", default=DEFAULT_TEXT_MODEL,
        help="LLM2Vec Llama-3 MNTP model ID/local directory. "
             "Uses the activated environment's Hugging Face cache.",
    )
    parser.add_argument("--text-revision")
    parser.add_argument("--text-adapter", help="LLM2Vec supervised adapter; defaults to the Kimodo Llama-3 adapter.")
    parser.add_argument("--text-adapter-revision")
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-fraction", type=float, default=1.0,
        help="Fraction of cached captioned train clips to use (0 < fraction <= 1); subset uses --seed. Val/test stay complete.",
    )
    parser.add_argument("--iterations", type=int, default=150000)
    parser.add_argument("--warmup-iterations", type=int, default=7500)
    parser.add_argument("--eval-interval", type=int, default=15000)
    parser.add_argument("--checkpoint-interval", type=int, default=15000)
    parser.add_argument("--log-freq", type=int, default=150)
    parser.add_argument("--start-lr", type=float, default=0.)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--validation-only", action="store_true", help="Do not evaluate test during hyperparameter search")
    parser.add_argument("--final-lr", type=float, default=1e-6,
        help="Final LR for the reference --lr; all optimizer groups use the same relative decay")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--use-bfloat16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
