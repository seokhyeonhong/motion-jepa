"""Standalone BABEL segmentation training for LoRA adapters plus a linear head.

Example (run from the repository root in the motion-jepa environment):
    python -m experiment.segmentation_probe.train_lora \
        --checkpoint /path/to/pretrained.pth.tar --output /path/to/lora-run \
        --rank 8 --alpha 16 --lora-lr 1e-4 --head-lr 1e-3

Repeat the same command with --resume to continue latest.pth.tar. Outputs are
compact adapter/head checkpoints, history.json, config.json and summary.json.
The original pretrained checkpoint is required for resume and inference; use
an immutable epoch checkpoint. load_lora_checkpoint() reconstructs a trained
model. Both module invocation and direct script invocation are supported.

This experiment updates the representation with supervised LoRA gradients.
Features are recomputed on every batch. Train-token channel statistics are
fitted once before adaptation and kept fixed. Best epoch selection uses
official BABEL-120 validation frame mAP (validation is not a held-out test).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.babel_segmentation import BabelSegmentationDataset
from experiment.linear_probe.features import (
    _atomic_json_save, _atomic_torch_save, _seed_all, _sha256_file,
    _torch_load_checkpoint, load_frozen_encoder, resolve_device,
    resolve_pretraining_stats,
)
from experiment.segmentation_probe.lora import DEFAULT_TARGETS, LoRASegmentationProbe
from experiment.segmentation_probe.metrics import FrameMetricAccumulator
from experiment.segmentation_probe.model import complete_patch_frame_mask
from experiment.segmentation_probe.online import (
    DEFAULT_DATASET, _capture_rng, _restore_rng,
)

FORMAT_VERSION = 1
KIND = "segmentation_lora"
STANDARDIZATION = "fixed_unadapted_encoder_all_complete_train_tokens_zscore"


def _path(value: str | Path) -> Path:
    value = Path(value).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _check_checkpoint(saved: dict) -> None:
    if saved.get("format_version") != FORMAT_VERSION or saved.get("kind") != KIND:
        raise ValueError("Unsupported LoRA segmentation checkpoint")


def _make_model(encoder, config: dict) -> LoRASegmentationProbe:
    return LoRASegmentationProbe(
        encoder, rank=config["rank"], alpha=config["alpha"],
        dropout=config["dropout"], targets=config["targets"])


def load_lora_checkpoint(path: str | Path, *, device: str = "auto",
                         pretrained_checkpoint: str | Path | None = None):
    """Return (eval-mode model, saved payload), checking the base weight hash.

    pretrained_checkpoint permits relocation of the original checkpoint; its
    bytes must match. Motion inputs must use the saved pretraining statistics.
    """
    saved = _torch_load_checkpoint(_path(path))
    _check_checkpoint(saved)
    config = saved["config"]
    base_path = _path(pretrained_checkpoint or config["checkpoint"])
    if _sha256_file(base_path) != saved["provenance"]["checkpoint_sha256"]:
        raise ValueError("Pretrained checkpoint differs from the LoRA run")
    encoder, _, info = load_frozen_encoder(
        base_path, config["checkpoint_key"], resolve_device(device))
    if info != saved["model_info"]:
        raise ValueError("Pretrained encoder geometry differs from the LoRA run")
    model = _make_model(encoder, config)
    model.load_adaptation_state_dict(saved["adaptation"])
    return model.eval(), saved


def make_loader(dataset, config: dict, device, *, shuffle: bool,
                generator: torch.Generator) -> DataLoader:
    return DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=shuffle,
        num_workers=config["num_workers"], pin_memory=device.type == "cuda",
        drop_last=False, persistent_workers=False, generator=generator)


def prepare_batch(batch, layout, device):
    motion, fps, lengths, labels, supervision, _ = batch
    motion = motion.to(device=device, dtype=torch.float32, non_blocking=True)
    fps = fps.to(device=device, dtype=torch.float32, non_blocking=True)
    lengths = lengths.to(device=device, dtype=torch.long, non_blocking=True)
    valid_frames = torch.arange(layout.raw_num_frames, device=device)[None] < lengths[:, None]
    if not layout.valid_token_mask(valid_frames).any(dim=1).all():
        raise ValueError("Every window must contain at least one complete encoder patch")
    complete = complete_patch_frame_mask(lengths, layout)
    frames = complete.shape[1]
    labels = labels[:, :frames].to(device=device, dtype=torch.float32, non_blocking=True)
    supervision = supervision[:, :frames].to(device=device, dtype=torch.bool, non_blocking=True)
    supervision = supervision & labels.bool().any(-1) & complete
    return motion, fps, valid_frames, labels, supervision


@torch.no_grad()
def fit_standardizer(model, loader, device, use_bfloat16: bool) -> dict:
    """Streaming FP64 moments over complete TRAIN tokens before adaptation."""
    model.eval()
    count, mean, m2 = 0, None, None
    for batch in tqdm(loader, desc="Fit train token statistics", mininterval=10):
        motion, fps, valid, _, _ = prepare_batch(batch, model.encoder.token_layout, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=use_bfloat16):
            tokens = model.encoder(motion, fps, valid_frames=valid)
        token_valid = model.encoder.token_layout.valid_token_mask(valid)
        values = tokens[token_valid].double()
        if not torch.isfinite(values).all():
            raise FloatingPointError("Non-finite tokens while fitting training statistics")
        batch_count = len(values)
        batch_mean = values.mean(0)
        batch_m2 = (values - batch_mean).square().sum(0)
        if mean is None:
            count, mean, m2 = batch_count, batch_mean, batch_m2
        else:
            total = count + batch_count
            delta = batch_mean - mean
            m2 += batch_m2 + delta.square() * (count * batch_count / total)
            mean += delta * (batch_count / total)
            count = total
    if not count:
        raise ValueError("No complete training tokens")
    model.feature_mean.copy_(mean.float())
    model.feature_scale.copy_((m2 / count).clamp_min(0).sqrt().clamp_min(1e-6).float())
    return {"policy": STANDARDIZATION, "fit_tokens": count, "epsilon": 1e-6,
            "fit_split": "train", "correction": 0}


def train_epoch(model, loader, optimizer, device, *, use_bfloat16: bool,
                gradient_clip: float) -> dict:
    model.train()
    loss_sum, frame_count, updates = 0.0, 0, 0
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for batch in tqdm(loader, desc="Train LoRA + head", mininterval=10):
        motion, fps, valid, labels, supervised = prepare_batch(
            batch, model.encoder.token_layout, device)
        count = int(supervised.sum())
        if not count:
            continue
        optimizer.zero_grad(set_to_none=True)
        logits = model(motion, fps, valid, use_bfloat16=use_bfloat16)
        loss = F.binary_cross_entropy_with_logits(logits[supervised], labels[supervised])
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite LoRA segmentation loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        loss_sum += float(loss.detach()) * count
        frame_count += count
        updates += 1
    if not frame_count:
        raise ValueError("No supervised training frames")
    if any(p.grad is not None for p in model.encoder.parameters() if not p.requires_grad):
        raise RuntimeError("Frozen encoder accumulated gradients")
    return {"bce": loss_sum / frame_count, "supervised_frames": frame_count,
            "updates": updates}


@torch.no_grad()
def evaluate(model, loader, device, *, use_bfloat16: bool, class_indices_60) -> dict:
    model.eval()
    # Store split-wide scores on CPU to bound GPU memory; AP still uses the
    # complete split with the existing tie-aware segmentation metric.
    metrics = FrameMetricAccumulator(120, class_indices_60, device="cpu")
    for batch in tqdm(loader, desc="Validate segmentation", mininterval=10):
        motion, fps, valid, labels, supervised = prepare_batch(
            batch, model.encoder.token_layout, device)
        logits = model(motion, fps, valid, use_bfloat16=use_bfloat16)
        metrics.update(logits, labels, supervised)
    return metrics.compute()


def _provenance(config: dict, dataset_root: Path, stats_root: Path) -> dict:
    names = ["meta.json", "index.json", "train.txt", "val.txt",
             "motions/train.json", "motions/val.json"]
    if (dataset_root / "class-index.json").is_file():
        names.append("class-index.json")
    return {
        "checkpoint_sha256": _sha256_file(Path(config["checkpoint"])),
        "checkpoint_key": config["checkpoint_key"],
        "dataset_hashes": {name: _sha256_file(dataset_root / name) for name in names},
        "stats_hashes": {name: _sha256_file(stats_root / name)
                        for name in ("mean.npy", "std.npy")},
        "standardization": STANDARDIZATION,
        "supervision": "positive_120_frames_and_complete_raw_patches",
        "selection": "babel-120/frame_map",
        "optimizer": "AdamW", "scheduler": "CosineAnnealingLR_eta_min_0",
    }


def _validate_config(config: dict) -> None:
    for key in ("epochs", "batch_size"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if config["num_workers"] < 0:
        raise ValueError("num_workers must be nonnegative")
    for key in ("lora_lr", "head_lr", "gradient_clip"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not math.isfinite(config["weight_decay"]) or config["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if config["precision"] not in ("float32", "bfloat16"):
        raise ValueError("precision must be float32 or bfloat16")


def train(config: dict, *, resume: bool = False,
          stop_after_epoch: int | None = None) -> dict:
    """Train a single-device experiment; stop_after_epoch is a resume-test hook."""
    config = dict(config)
    _validate_config(config)
    for key in ("checkpoint", "dataset_root", "output"):
        config[key] = str(_path(config[key]))
    if config.get("stats_root") is not None:
        config["stats_root"] = str(_path(config["stats_root"]))
    config["targets"] = list(config["targets"])
    output = Path(config["output"])
    if resume:
        if not (output / "latest.pth.tar").is_file():
            raise FileNotFoundError("Resume requires <output>/latest.pth.tar")
        saved = _torch_load_checkpoint(output / "latest.pth.tar")
        _check_checkpoint(saved)
        # Workers/device may change without changing the optimization protocol.
        for key, value in config.items():
            if key not in {"device", "num_workers", "output"} and saved["config"].get(key) != value:
                raise ValueError(f"Resume requires unchanged {key}")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Output is nonempty: {output}; use --resume or a new --output")
        saved = None
    device = resolve_device(config["device"])
    use_bfloat16 = config["precision"] == "bfloat16"
    if use_bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This CUDA device does not support BF16")
    _seed_all(config["seed"])
    encoder, pretraining_config, model_info = load_frozen_encoder(
        Path(config["checkpoint"]), config["checkpoint_key"], device)
    if encoder.token_layout.kind != "1d":
        raise ValueError("LoRA segmentation requires a 1D pretrained encoder")
    stats_root = resolve_pretraining_stats(
        pretraining_config, Path(config["stats_root"]) if config.get("stats_root") else None)
    dataset_root = Path(config["dataset_root"])
    datasets = {
        split: BabelSegmentationDataset(
            dataset_root, split, num_frames=model_info["num_frames"],
            fps=model_info["fps"], motion_dim=model_info["motion_dim"],
            stats_root=stats_root, normalize=True)
        for split in ("train", "val")
    }
    if any(not len(dataset) for dataset in datasets.values()):
        raise ValueError("LoRA segmentation requires nonempty train and validation splits")
    if (datasets["train"].class_names != datasets["val"].class_names
            or datasets["train"].class_indices_60 != datasets["val"].class_indices_60):
        raise ValueError("Training and validation class vocabularies differ")
    provenance = _provenance(config, dataset_root, stats_root)
    if saved is not None and (saved["provenance"] != provenance or saved["model_info"] != model_info):
        raise ValueError("Resume checkpoint/dataset/statistics provenance differs")
    model = _make_model(encoder, config)
    optimizer = torch.optim.AdamW([
        {"params": list(model.adapter_parameters()), "lr": config["lora_lr"], "name": "lora"},
        {"params": list(model.head.parameters()), "lr": config["head_lr"], "name": "head"},
    ], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    generators = {split: torch.Generator().manual_seed(config["seed"] + index)
                  for index, split in enumerate(("train", "val", "statistics"))}
    loaders = {split: make_loader(
        datasets[split], config, device, shuffle=split == "train", generator=generators[split])
        for split in ("train", "val")}
    first_epoch, best_score, best_epoch, best_val, history = 1, -math.inf, None, None, []
    if saved is not None:
        model.load_adaptation_state_dict(saved["adaptation"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        first_epoch = saved["next_epoch"]
        best_score, best_epoch, best_val = saved["best_score"], saved["best_epoch"], saved["best_val"]
        history, normalizer = saved["history"], saved["normalizer"]
        best_path = output / "best.pth.tar"
        if not best_path.is_file():
            raise FileNotFoundError("Resume requires the paired best.pth.tar")
        best_saved = _torch_load_checkpoint(best_path)
        _check_checkpoint(best_saved)
        if best_saved["provenance"] != provenance or best_saved["best_score"] < best_score:
            raise ValueError("Saved best checkpoint does not match this run")
        _restore_rng(saved["rng"])
        for split, generator in generators.items():
            generator.set_state(saved["loader_rng"][split])
    else:
        statistics_loader = make_loader(
            datasets["train"], config, device, shuffle=False, generator=generators["statistics"])
        normalizer = fit_standardizer(model, statistics_loader, device, use_bfloat16)
    output.mkdir(parents=True, exist_ok=True)
    parameter_counts = {
        "lora": sum(p.numel() for p in model.adapter_parameters()),
        "head": sum(p.numel() for p in model.head.parameters()),
        "frozen_encoder": sum(p.numel() for p in encoder.parameters() if not p.requires_grad),
    }
    _atomic_json_save(
        {"config": config, "provenance": provenance, "model_info": model_info,
         "adapter_names": model.adapter_names, "parameter_counts": parameter_counts,
         "stats_root": str(stats_root), "normalizer": normalizer,
         "class_names": datasets["train"].class_names,
         "class_indices_60": datasets["train"].class_indices_60}, output / "config.json")
    print(json.dumps({"event": "start", "parameters": parameter_counts,
                      "first_epoch": first_epoch, "epochs": config["epochs"],
                      "output": str(output)}), flush=True)
    for epoch in range(first_epoch, config["epochs"] + 1):
        started = time.perf_counter()
        rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
        training = train_epoch(
            model, loaders["train"], optimizer, device, use_bfloat16=use_bfloat16,
            gradient_clip=config["gradient_clip"])
        validation = evaluate(
            model, loaders["val"], device, use_bfloat16=use_bfloat16,
            class_indices_60=datasets["train"].class_indices_60)
        score = validation["babel-120"]["frame_map"]
        if score is None or not math.isfinite(score):
            raise ValueError("Validation has no finite BABEL-120 frame mAP")
        improved = score > best_score
        if improved:
            best_score, best_epoch, best_val = score, epoch, validation
        row = {"epoch": epoch, "train": training, "val": validation,
               "lr": rates, "seconds": time.perf_counter() - started}
        history.append(row)
        scheduler.step()
        payload = {
            "format_version": FORMAT_VERSION, "kind": KIND,
            "config": config, "provenance": provenance, "model_info": model_info,
            "adapter_names": model.adapter_names, "parameter_counts": parameter_counts,
            "stats_root": str(stats_root), "normalizer": normalizer,
            "class_names": datasets["train"].class_names,
            "class_indices_60": datasets["train"].class_indices_60,
            "adaptation": model.adaptation_state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "next_epoch": epoch + 1, "best_score": best_score, "best_epoch": best_epoch,
            "best_val": best_val, "history": history, "rng": _capture_rng(),
            "loader_rng": {name: generator.get_state() for name, generator in generators.items()},
        }
        if improved:
            _atomic_torch_save(payload, output / "best.pth.tar")
        _atomic_torch_save(payload, output / "latest.pth.tar")
        _atomic_json_save(history, output / "history.json")
        print(json.dumps({"event": "epoch", **row, "best_epoch": best_epoch}), flush=True)
        if stop_after_epoch is not None and epoch >= stop_after_epoch and epoch < config["epochs"]:
            return {"complete": False, "history": history, "output": str(output)}
    summary = {
        "complete": True, "kind": KIND, "config": config, "provenance": provenance,
        "best_epoch": best_epoch, "best_val": best_val, "test": None,
        "selection": "babel-120/frame_map", "parameter_counts": parameter_counts,
        "normalizer": normalizer, "best_path": str(output / "best.pth.tar"),
        "latest_path": str(output / "latest.pth.tar"), "history_path": str(output / "history.json"),
    }
    _atomic_json_save(summary, output / "summary.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Immutable MotionJEPA pretraining checkpoint")
    parser.add_argument("--checkpoint-key", choices=("target_encoder", "encoder"), default="target_encoder")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET)
    parser.add_argument("--stats-root", default=None, help="Override pretraining motion statistics directory")
    parser.add_argument("--output", required=True, help="New experiment directory")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0, help="Adapter scale is alpha / rank")
    parser.add_argument("--dropout", type=float, default=0.0, help="LoRA input dropout only")
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS),
                        help="Linear module suffixes (default: attn.qkv attn.proj); qkv adapts Q, K and V")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--resume", action="store_true",
                        help="Resume output/latest.pth.tar with the same training arguments")
    return parser


def main():
    args = vars(build_parser().parse_args())
    resume = args.pop("resume")
    summary = train(args, resume=resume)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
