"""CNN and Transformer classification from raw motion or frozen JEPA tokens."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from .cnn import MotionCNNClassifier
from .dataset import StyleTokenDataset, build_style_datasets, load_style_label_index
from .features import (
    Metrics,
    _atomic_json_save,
    _atomic_torch_save,
    _seed_all,
    _sha256_file,
    _torch_load_checkpoint,
    load_frozen_encoder,
    resolve_pretraining_stats,
    resolve_device,
)
from .transformer import MotionTransformerClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_FINDINGS_ROOT = (
    PROJECT_ROOT / "findings/000-100style-classification/classifiers"
)
MODELS = ("cnn", "transformer")
METRIC_FIELDS = (
    "loss",
    "top1_accuracy",
    "macro_accuracy",
    "top5_accuracy",
)
TOKEN_CACHE_FORMAT_VERSION = 1
CLASSIFIER_CHECKPOINT_FORMAT_VERSION = 2


@dataclass(frozen=True)
class PreparedInput:
    datasets: dict[str, Any]
    label_index: Any
    input_dim: int
    num_frames: int
    stats_root: Path
    input_source: str
    jepa_source: dict[str, Any] | None


class MetricAccumulator:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = int(num_classes)
        self.loss_sum = 0.0
        self.total = 0
        self.correct = 0
        self.top5_correct = 0
        self.class_total = torch.zeros(self.num_classes, dtype=torch.long)
        self.class_correct = torch.zeros(self.num_classes, dtype=torch.long)

    def update(
        self, logits: torch.Tensor, labels: torch.Tensor, loss_sum: float
    ) -> None:
        predictions = logits.detach().argmax(dim=1)
        labels = labels.detach()
        matches = predictions.eq(labels)
        self.loss_sum += float(loss_sum)
        self.total += len(labels)
        self.correct += int(matches.sum())
        topk = min(5, self.num_classes)
        self.top5_correct += int(
            logits.detach()
            .topk(topk, dim=1)
            .indices.eq(labels[:, None])
            .any(dim=1)
            .sum()
        )
        cpu_labels = labels.cpu()
        self.class_total += torch.bincount(cpu_labels, minlength=self.num_classes)
        self.class_correct += torch.bincount(
            cpu_labels[matches.cpu()], minlength=self.num_classes
        )

    def compute(self) -> Metrics:
        if self.total == 0:
            raise ValueError("Cannot compute metrics for an empty split")
        present = self.class_total > 0
        macro = (
            self.class_correct[present].float() / self.class_total[present].float()
        ).mean()
        return Metrics(
            loss=self.loss_sum / self.total,
            top1_accuracy=self.correct / self.total,
            macro_accuracy=float(macro),
            top5_accuracy=self.top5_correct / self.total,
        )


def make_classifier(
    model_name: str,
    *,
    motion_dim: int | None = None,
    input_dim: int | None = None,
    num_frames: int,
    num_classes: int,
) -> tuple[nn.Module, dict[str, Any]]:
    if motion_dim is not None and input_dim is not None and motion_dim != input_dim:
        raise ValueError("motion_dim and input_dim must match when both are provided")
    resolved_input_dim = int(
        input_dim if input_dim is not None else motion_dim if motion_dim is not None else 366
    )
    if model_name == "cnn":
        config: dict[str, Any] = {
            "name": "MotionCNNClassifier",
            "input_dim": resolved_input_dim,
            "num_classes": num_classes,
            "widths": [256, 384, 512],
            "blocks_per_stage": 2,
            "dropout": 0.1,
        }
        model = MotionCNNClassifier(
            input_dim=resolved_input_dim,
            num_classes=num_classes,
            widths=tuple(config["widths"]),
            blocks_per_stage=config["blocks_per_stage"],
            dropout=config["dropout"],
        )
    elif model_name == "transformer":
        config = {
            "name": "MotionTransformerClassifier",
            "input_dim": resolved_input_dim,
            "num_frames": num_frames,
            "num_classes": num_classes,
            "embed_dim": 256,
            "depth": 8,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
            "drop_path_rate": 0.1,
        }
        model = MotionTransformerClassifier(
            **{key: value for key, value in config.items() if key != "name"}
        )
    else:
        raise ValueError(f"Unknown classifier model: {model_name}")
    return model, config


def _capture_rng_state(generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_generator": generator.get_state(),
    }


def _restore_rng_state(state: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader_generator"])


def _lr_factor(
    epoch: int, *, epochs: int, warmup_epochs: int, final_factor: float
) -> float:
    if warmup_epochs and epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    cosine_epochs = epochs - warmup_epochs
    if cosine_epochs <= 1:
        return final_factor
    progress = (epoch - warmup_epochs) / (cosine_epochs - 1)
    progress = min(max(progress, 0.0), 1.0)
    return final_factor + (1.0 - final_factor) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def _amp_context(device: torch.device, use_bfloat16: bool):
    if device.type == "cuda" and use_bfloat16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _valid_frames(motion: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    return (
        torch.arange(motion.shape[1], device=motion.device).unsqueeze(0)
        < length.unsqueeze(1)
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    num_classes: int,
    use_bfloat16: bool,
    gradient_clip: float,
) -> Metrics:
    model.train()
    metrics = MetricAccumulator(num_classes)
    for motion, _, length, labels, _ in loader:
        motion = motion.to(device=device, dtype=torch.float32, non_blocking=True)
        length = length.to(device=device, non_blocking=True)
        labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
        active = _valid_frames(motion, length)
        optimizer.zero_grad(set_to_none=True)
        with _amp_context(device, use_bfloat16):
            logits = model(motion, active)
            loss = F.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        metrics.update(logits, labels, float(loss.detach()) * len(labels))
    return metrics.compute()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    num_classes: int,
    use_bfloat16: bool,
) -> Metrics:
    model.eval()
    metrics = MetricAccumulator(num_classes)
    with torch.inference_mode():
        for motion, _, length, labels, _ in loader:
            motion = motion.to(device=device, dtype=torch.float32, non_blocking=True)
            length = length.to(device=device, non_blocking=True)
            labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
            with _amp_context(device, use_bfloat16):
                logits = model(motion, _valid_frames(motion, length))
                loss_sum = F.cross_entropy(logits, labels, reduction="sum")
            metrics.update(logits, labels, float(loss_sum))
    return metrics.compute()


def _read_dataset_meta(dataset_root: Path) -> dict[str, Any]:
    path = dataset_root / "meta.json"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset metadata does not exist: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    return {
        "num_frames": int(metadata["num_frames"]),
        "fps": int(metadata["fps"]),
        "motion_dim": int(metadata["motion_dim"]),
    }


def _argument(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _token_cache_metadata(
    *,
    split: str,
    dataset_root: Path,
    stats_root: Path,
    checkpoint_path: Path,
    checkpoint_key: str,
    model_info: dict[str, Any],
    class_names: list[str],
) -> dict[str, Any]:
    return {
        "format_version": TOKEN_CACHE_FORMAT_VERSION,
        "kind": "motion_jepa_frame_tokens",
        "split": split,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_key": checkpoint_key,
        "dataset_index_sha256": _sha256_file(dataset_root / "index.json"),
        "stats_mean_sha256": _sha256_file(stats_root / "mean.npy"),
        "stats_std_sha256": _sha256_file(stats_root / "std.npy"),
        "model_name": model_info["model_name"],
        "num_frames": model_info["num_frames"],
        "motion_dim": model_info["motion_dim"],
        "fps": model_info["fps"],
        "feature_dim": model_info["feature_dim"],
        "token_num_frames": model_info["token_num_frames"],
        "temporal_patch_size": model_info["temporal_patch_size"],
        "patchified": model_info["patchified"],
        "layout_kind": model_info["kind"],
        "dtype": "bfloat16",
        "class_names": class_names,
    }


def _validate_token_cache(
    payload: dict[str, Any],
    expected_metadata: dict[str, Any],
    *,
    label_index: Any,
) -> None:
    if payload.get("metadata") != expected_metadata:
        raise ValueError(
            "JEPA token cache metadata is stale; rerun with --recompute-features"
        )
    dataset = StyleTokenDataset(
        payload,
        label_index=label_index,
        fps=int(expected_metadata["fps"]),
    )
    expected_shape = (
        int(expected_metadata["token_num_frames"]),
        int(expected_metadata["feature_dim"]),
    )
    if dataset.features.shape[1:] != expected_shape:
        raise ValueError(
            f"Token cache feature shape must end in {expected_shape}, "
            f"got {tuple(dataset.features.shape)}"
        )


def _extract_token_features(
    encoder: nn.Module,
    dataset: Any,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    feature_batches: list[torch.Tensor] = []
    length_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    sample_ids: list[str] = []
    with torch.inference_mode():
        for motion, fps, length, labels, ids in tqdm(
            loader, desc="Extract JEPA tokens"
        ):
            motion = motion.to(device=device, dtype=torch.float32, non_blocking=True)
            fps = fps.to(device=device, dtype=torch.float32, non_blocking=True)
            length_device = length.to(device=device, dtype=torch.long, non_blocking=True)
            active = _valid_frames(motion, length_device)
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with amp_context:
                encoded = encoder(motion, fps, valid_frames=active)
            if encoded.ndim != 3:
                raise ValueError(
                    "JEPA token classifiers require a 1D encoder output [B,T,D], "
                    f"got {tuple(encoded.shape)}"
                )
            token_active = encoder.token_layout.valid_token_mask(active)
            encoded = encoded * token_active.unsqueeze(-1).to(dtype=encoded.dtype)
            feature_batches.append(encoded.to(dtype=torch.bfloat16).cpu())
            token_lengths = encoder.token_layout.valid_token_lengths(length_device)
            length_batches.append(token_lengths.to(dtype=torch.long).cpu())
            label_batches.append(labels.to(dtype=torch.long).cpu())
            sample_ids.extend(list(ids))
    if not feature_batches:
        raise ValueError("Cannot extract JEPA tokens from an empty split")
    return {
        "features": torch.cat(feature_batches),
        "lengths": torch.cat(length_batches),
        "labels": torch.cat(label_batches),
        "sample_ids": sample_ids,
    }


def _load_or_extract_token_split(
    *,
    cache_path: Path,
    metadata: dict[str, Any],
    label_index: Any,
    dataset: Any,
    encoder: nn.Module,
    device: torch.device,
    feature_batch_size: int,
    num_workers: int,
    recompute: bool,
) -> dict[str, Any]:
    if cache_path.is_file() and not recompute:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        _validate_token_cache(payload, metadata, label_index=label_index)
        return payload
    payload = _extract_token_features(
        encoder,
        dataset,
        device=device,
        batch_size=feature_batch_size,
        num_workers=num_workers,
    )
    payload["metadata"] = metadata
    _validate_token_cache(payload, metadata, label_index=label_index)
    _atomic_torch_save(payload, cache_path)
    return payload


def _prepare_input(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> PreparedInput:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    dataset_info = _read_dataset_meta(dataset_root)
    input_source = str(_argument(args, "input_source", "raw"))
    if input_source == "raw":
        stats_root = dataset_root / "stats"
        datasets, label_index = build_style_datasets(
            dataset_root,
            num_frames=dataset_info["num_frames"],
            fps=dataset_info["fps"],
            motion_dim=dataset_info["motion_dim"],
            stats_root=stats_root,
        )
        return PreparedInput(
            datasets=datasets,
            label_index=label_index,
            input_dim=dataset_info["motion_dim"],
            num_frames=dataset_info["num_frames"],
            stats_root=stats_root.resolve(),
            input_source="raw",
            jepa_source=None,
        )
    if input_source != "jepa":
        raise ValueError(f"Unknown input source: {input_source}")
    checkpoint_value = _argument(args, "jepa_checkpoint", None)
    if checkpoint_value is None:
        raise ValueError("--jepa-checkpoint is required for --input-source jepa")
    checkpoint_path = Path(checkpoint_value).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"JEPA checkpoint does not exist: {checkpoint_path}")
    checkpoint_key = str(_argument(args, "checkpoint_key", "target_encoder"))
    encoder, config, model_info = load_frozen_encoder(
        checkpoint_path, checkpoint_key, device
    )
    if model_info["kind"] != "1d":
        raise ValueError("JEPA token classifiers currently support only 1D encoders")
    for key in ("num_frames", "motion_dim", "fps"):
        if int(model_info[key]) != int(dataset_info[key]):
            raise ValueError(
                f"JEPA checkpoint {key}={model_info[key]} does not match "
                f"dataset {key}={dataset_info[key]}"
            )
    stats_value = _argument(args, "stats_path", None)
    stats_root = resolve_pretraining_stats(
        config, Path(stats_value) if stats_value is not None else None
    )
    label_index = load_style_label_index(dataset_root)
    motion_datasets, _ = build_style_datasets(
        dataset_root,
        num_frames=model_info["num_frames"],
        fps=model_info["fps"],
        motion_dim=model_info["motion_dim"],
        stats_root=stats_root,
        label_index=label_index,
    )
    cache_value = _argument(args, "feature_cache_root", None)
    cache_root = (
        Path(cache_value).expanduser().resolve()
        if cache_value is not None
        else checkpoint_path.parent / "linear-probe/token-features"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    token_datasets: dict[str, StyleTokenDataset] = {}
    feature_batch_size = int(_argument(args, "feature_batch_size", 256))
    class_names = list(label_index.class_names)
    for split in ("train", "val", "test"):
        metadata = _token_cache_metadata(
            split=split,
            dataset_root=dataset_root,
            stats_root=stats_root,
            checkpoint_path=checkpoint_path,
            checkpoint_key=checkpoint_key,
            model_info=model_info,
            class_names=class_names,
        )
        payload = _load_or_extract_token_split(
            cache_path=cache_root / f"{split}.pt",
            metadata=metadata,
            label_index=label_index,
            dataset=motion_datasets[split],
            encoder=encoder,
            device=device,
            feature_batch_size=feature_batch_size,
            num_workers=args.num_workers,
            recompute=bool(_argument(args, "recompute_features", False)),
        )
        token_datasets[split] = StyleTokenDataset(
            payload, label_index=label_index, fps=model_info["fps"]
        )
    jepa_source = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_key": checkpoint_key,
        "model_name": model_info["model_name"],
        "feature_dim": model_info["feature_dim"],
        "num_frames": model_info["num_frames"],
        "token_num_frames": model_info["token_num_frames"],
        "temporal_patch_size": model_info["temporal_patch_size"],
        "stats_root": str(stats_root),
        "stats_mean_sha256": _sha256_file(stats_root / "mean.npy"),
        "stats_std_sha256": _sha256_file(stats_root / "std.npy"),
        "token_cache_root": str(cache_root),
        "token_cache_dtype": "bfloat16",
        "token_cache_format_version": TOKEN_CACHE_FORMAT_VERSION,
    }
    del encoder, motion_datasets
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return PreparedInput(
        datasets=token_datasets,
        label_index=label_index,
        input_dim=int(model_info["feature_dim"]),
        num_frames=int(model_info["token_num_frames"]),
        stats_root=stats_root,
        input_source="jepa",
        jepa_source=jepa_source,
    )


def _make_loaders(
    datasets: dict[str, Any],
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, DataLoader]:
    return {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            generator=generator if split == "train" else None,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
            drop_last=False,
        )
        for split, dataset in datasets.items()
    }


def _signature(
    args: argparse.Namespace,
    model_name: str,
    dataset_root: Path,
    model_config: dict[str, Any],
    prepared: PreparedInput,
) -> dict[str, Any]:
    signature = {
        "model": model_name,
        "architecture": model_config,
        "dataset_index_sha256": _sha256_file(dataset_root / "index.json"),
        "stats_mean_sha256": _sha256_file(prepared.stats_root / "mean.npy"),
        "stats_std_sha256": _sha256_file(prepared.stats_root / "std.npy"),
        "input_source": prepared.input_source,
        "seed": args.seed,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "final_lr": args.final_lr,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "use_bfloat16": args.use_bfloat16,
    }
    if prepared.jepa_source is not None:
        signature["jepa_source"] = prepared.jepa_source
    return signature


def _result_fields() -> list[str]:
    return [
        "epoch",
        "learning_rate",
        *[f"train_{field}" for field in METRIC_FIELDS],
        *[f"val_{field}" for field in METRIC_FIELDS],
    ]


def run_model(
    args: argparse.Namespace,
    model_name: str,
    *,
    prepared: PreparedInput,
    output_root: Path,
    device: torch.device | None = None,
) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output = output_root / model_name / f"seed-{args.seed}"
    device = resolve_device(args.device) if device is None else device
    datasets = prepared.datasets
    label_index = prepared.label_index
    train_labels = {int(label) for label in datasets["train"].labels}
    missing_train = set(range(label_index.num_classes)) - train_labels
    if missing_train:
        raise ValueError(f"Training split is missing class IDs: {sorted(missing_train)}")
    _seed_all(args.seed)
    model, model_config = make_classifier(
        model_name,
        input_dim=prepared.input_dim,
        num_frames=prepared.num_frames,
        num_classes=label_index.num_classes,
    )
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    signature = _signature(
        args, model_name, dataset_root, model_config, prepared
    )
    summary_path = output / "summary.json"
    if summary_path.is_file() and not args.overwrite:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") == "complete" and summary.get("signature") == signature:
            return summary
        raise FileExistsError(f"Classifier result already exists with another config: {output}")
    output.mkdir(parents=True, exist_ok=True)
    existing = [
        name
        for name in (
            "metrics.csv",
            "classifier-best.pth.tar",
            "classifier-latest.pth.tar",
        )
        if (output / name).exists()
    ]
    if existing and not (args.resume or args.overwrite):
        raise FileExistsError(
            f"Partial classifier output exists under {output}: {existing}; use --resume"
        )
    if args.overwrite:
        for name in existing + ["summary.json", "model-config.json"]:
            path = output / name
            if path.exists():
                path.unlink()

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    final_factor = args.final_lr / args.lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: _lr_factor(
            epoch,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            final_factor=final_factor,
        ),
    )
    generator = torch.Generator().manual_seed(args.seed)
    start_epoch = 0
    best_epoch = 0
    best_accuracy = -1.0
    latest_path = output / "classifier-latest.pth.tar"
    best_path = output / "classifier-best.pth.tar"
    if args.resume and latest_path.is_file():
        checkpoint = _torch_load_checkpoint(latest_path)
        if checkpoint.get("format_version") != CLASSIFIER_CHECKPOINT_FORMAT_VERSION:
            raise ValueError("Unsupported latest classifier checkpoint format")
        if checkpoint.get("signature") != signature:
            raise ValueError("Latest classifier checkpoint config does not match this run")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["next_epoch"])
        best_epoch = int(checkpoint["best_epoch"])
        best_accuracy = float(checkpoint["best_accuracy"])
        _restore_rng_state(checkpoint["rng_state"], generator)
    loaders = _make_loaders(
        datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        generator=generator,
    )
    _atomic_json_save(label_index.to_json(), output / "class-index.json")

    metrics_path = output / "metrics.csv"
    fields = _result_fields()
    mode = "a" if start_epoch else "w"
    if start_epoch:
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Resume metrics do not exist: {metrics_path}")
        with metrics_path.open(encoding="utf-8", newline="") as file:
            if len(list(csv.DictReader(file))) != start_epoch:
                raise ValueError("Metrics row count does not match checkpoint next_epoch")
    with metrics_path.open(mode, encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if not start_epoch:
            writer.writeheader()
        for epoch in range(start_epoch, args.epochs):
            current_lr = float(optimizer.param_groups[0]["lr"])
            train_metrics = train_epoch(
                model,
                loaders["train"],
                optimizer,
                device=device,
                num_classes=label_index.num_classes,
                use_bfloat16=args.use_bfloat16,
                gradient_clip=args.gradient_clip,
            )
            val_metrics = evaluate(
                model,
                loaders["val"],
                device=device,
                num_classes=label_index.num_classes,
                use_bfloat16=args.use_bfloat16,
            )
            row = {
                "epoch": epoch + 1,
                "learning_rate": current_lr,
                **{f"train_{key}": value for key, value in asdict(train_metrics).items()},
                **{f"val_{key}": value for key, value in asdict(val_metrics).items()},
            }
            writer.writerow(row)
            file.flush()
            if val_metrics.top1_accuracy > best_accuracy:
                best_accuracy = val_metrics.top1_accuracy
                best_epoch = epoch + 1
                _atomic_torch_save(
                    {
                        "format_version": CLASSIFIER_CHECKPOINT_FORMAT_VERSION,
                        "model": model.state_dict(),
                        "architecture": model_config,
                    },
                    best_path,
                )
            scheduler.step()
            _atomic_torch_save(
                {
                    "format_version": CLASSIFIER_CHECKPOINT_FORMAT_VERSION,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "next_epoch": epoch + 1,
                    "best_epoch": best_epoch,
                    "best_accuracy": best_accuracy,
                    "rng_state": _capture_rng_state(generator),
                    "signature": signature,
                },
                latest_path,
            )
            print(
                f"{model_name} epoch {epoch + 1:03d}/{args.epochs}: "
                f"train={train_metrics.top1_accuracy:.4f}, "
                f"val={val_metrics.top1_accuracy:.4f}, lr={current_lr:.3e}",
                flush=True,
            )

    best = _torch_load_checkpoint(best_path)
    if best.get("format_version") != CLASSIFIER_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported best classifier checkpoint format")
    model.load_state_dict(best["model"], strict=True)
    best_val_metrics = evaluate(
        model,
        loaders["val"],
        device=device,
        num_classes=label_index.num_classes,
        use_bfloat16=args.use_bfloat16,
    )
    test_metrics = evaluate(
        model,
        loaders["test"],
        device=device,
        num_classes=label_index.num_classes,
        use_bfloat16=args.use_bfloat16,
    )
    summary = {
        "status": "complete",
        "model": model_name,
        "num_parameters": num_parameters,
        "best_epoch": best_epoch,
        "best_val": asdict(best_val_metrics),
        "test": asdict(test_metrics),
        "split_counts": {split: len(dataset) for split, dataset in datasets.items()},
        "signature": signature,
    }
    _atomic_json_save(summary, summary_path)
    latest_path.unlink(missing_ok=True)
    (output / "model-config.json").unlink(missing_ok=True)
    return summary


def _copy_metrics(output_root: Path, findings_root: Path, model_name: str, seed: int) -> Path:
    source = output_root / model_name / f"seed-{seed}" / "metrics.csv"
    destination = findings_root / f"{model_name}-metrics.csv"
    shutil.copyfile(source, destination)
    return destination


def write_findings(
    summaries: dict[str, dict[str, Any]],
    *,
    output_root: Path,
    findings_root: Path,
    seed: int,
    input_source: str,
    jepa_source: dict[str, Any] | None,
) -> None:
    findings_root.mkdir(parents=True, exist_ok=True)
    _atomic_json_save(
        {
            "format_version": 1,
            "input_source": input_source,
            "seed": seed,
            "summaries": summaries,
        },
        findings_root / "results.json",
    )
    metrics: dict[str, list[dict[str, str]]] = {}
    for model_name in MODELS:
        path = _copy_metrics(output_root, findings_root, model_name, seed)
        with path.open(encoding="utf-8", newline="") as file:
            metrics[model_name] = list(csv.DictReader(file))

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for model_name, color in (("cnn", "tab:blue"), ("transformer", "tab:orange")):
        rows = metrics[model_name]
        epochs = [int(row["epoch"]) for row in rows]
        axes[0, 0].plot(epochs, [float(row["train_loss"]) for row in rows], label=model_name, color=color)
        axes[0, 1].plot(epochs, [float(row["val_loss"]) for row in rows], label=model_name, color=color)
        axes[1, 0].plot(epochs, [float(row["train_top1_accuracy"]) * 100 for row in rows], label=model_name, color=color)
        axes[1, 1].plot(epochs, [float(row["val_top1_accuracy"]) * 100 for row in rows], label=model_name, color=color)
    titles = (("Train loss", "Validation loss"), ("Train top-1 (%)", "Validation top-1 (%)"))
    for row_index, row_axes in enumerate(axes):
        for column_index, axis in enumerate(row_axes):
            axis.set_title(titles[row_index][column_index])
            axis.grid(True, alpha=0.25)
            axis.legend()
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 1].set_xlabel("Epoch")
    figure.tight_layout()
    figure.savefig(findings_root / "training-curves.png", dpi=180)
    plt.close(figure)

    source_description = (
        "100STYLE raw motion `[90,366]`"
        if input_source == "raw"
        else (
            f"frozen `{jepa_source['model_name']}` frame-token features "
            f"`[90,{jepa_source['feature_dim']}]`"
        )
    )
    lines = [
        "# 100STYLE Raw-Motion Classifiers",
        "",
        f"Both models were trained from {source_description} with seed {seed}.",
        "",
        "## Shared settings",
        "",
        (
            "- 100STYLE train statistics normalization"
            if input_source == "raw"
            else "- Frozen JEPA target-encoder tokens using pretraining statistics"
        ),
        "- CrossEntropyLoss; no class weighting, balanced sampling, or augmentation",
        "- AdamW, LR 3e-4, weight decay 0.05",
        "- 5-epoch warmup followed by cosine decay, 100 epochs",
        "- CUDA BF16 autocast; float32 parameters and optimizer",
        "- Best validation top-1 checkpoint restored before one test evaluation",
        "",
        "![Training curves](training-curves.png)",
        "",
        "## Results",
        "",
        "| Model | Parameters | Best epoch | Val top-1 | Val macro | Val top-5 | Test top-1 | Test macro | Test top-5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if jepa_source is not None:
        lines[3:3] = [
            "",
            f"- JEPA checkpoint: `{jepa_source['checkpoint_path']}`",
            f"- SHA256: `{jepa_source['checkpoint_sha256']}`",
            f"- Checkpoint key: `{jepa_source['checkpoint_key']}`",
        ]
    for model_name in MODELS:
        summary = summaries[model_name]
        val = summary["best_val"]
        test = summary["test"]
        lines.append(
            f"| {model_name} | {summary['num_parameters']:,} | {summary['best_epoch']} | "
            f"{val['top1_accuracy'] * 100:.2f} | {val['macro_accuracy'] * 100:.2f} | "
            f"{val['top5_accuracy'] * 100:.2f} | {test['top1_accuracy'] * 100:.2f} | "
            f"{test['macro_accuracy'] * 100:.2f} | {test['top5_accuracy'] * 100:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Model architecture",
            "",
            "- CNN: temporal ResNet, widths 256/384/512, two blocks per stage, masked mean pooling.",
            "- Transformer: dim 256, 8 blocks, 8 heads, MLP ratio 4, learnable CLS-token pooling.",
            "",
            "## Raw metrics",
            "",
            "- [CNN metrics](cnn-metrics.csv)",
            "- [Transformer metrics](transformer-metrics.csv)",
            "- [Classifier summaries](results.json)",
            "",
        ]
    )
    (findings_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    if args.epochs <= 0 or not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("epochs must be positive and warmup_epochs must be in [0, epochs)")
    if args.lr <= 0 or args.final_lr < 0 or args.final_lr > args.lr:
        raise ValueError("Require 0 <= final_lr <= lr and lr > 0")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if int(_argument(args, "feature_batch_size", 256)) <= 0:
        raise ValueError("feature_batch_size must be positive")
    device = resolve_device(args.device)
    input_source = str(_argument(args, "input_source", "raw"))
    checkpoint_value = _argument(args, "jepa_checkpoint", None)
    output_value = _argument(args, "output_root", None)
    findings_value = _argument(args, "findings_root", None)
    if input_source == "jepa":
        if checkpoint_value is None:
            raise ValueError("--jepa-checkpoint is required for --input-source jepa")
        checkpoint_root = Path(checkpoint_value).expanduser().resolve().parent
        default_output = checkpoint_root / "linear-probe/classifiers"
        default_findings = default_output / "findings"
    else:
        default_output = PROJECT_ROOT / "output/100style-classifiers"
        default_findings = DEFAULT_RAW_FINDINGS_ROOT
    output_root = (
        Path(output_value).expanduser().resolve()
        if output_value is not None
        else default_output
    )
    findings_root = (
        Path(findings_value).expanduser().resolve()
        if findings_value is not None
        else default_findings
    )
    prepared = _prepare_input(args, device=device)
    selected = MODELS if args.model == "all" else (args.model,)
    summaries = {
        model_name: run_model(
            args,
            model_name,
            prepared=prepared,
            output_root=output_root,
            device=device,
        )
        for model_name in selected
    }
    if set(summaries) == set(MODELS):
        write_findings(
            summaries,
            output_root=output_root,
            findings_root=findings_root,
            seed=args.seed,
            input_source=prepared.input_source,
            jepa_source=prepared.jepa_source,
        )
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train 100STYLE classifiers from raw motion or frozen JEPA tokens"
    )
    parser.add_argument("--model", choices=(*MODELS, "all"), default="all")
    parser.add_argument("--input-source", choices=("raw", "jepa"), default="raw")
    parser.add_argument("--jepa-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-key",
        choices=("target_encoder", "encoder"),
        default="target_encoder",
    )
    parser.add_argument("--stats-path", type=Path, default=None)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--feature-cache-root", type=Path, default=None)
    parser.add_argument("--recompute-features", action="store_true")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "dataset/100style-soma77-processed",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: raw output root or <JEPA checkpoint dir>/linear-probe/classifiers",
    )
    parser.add_argument(
        "--findings-root",
        type=Path,
        default=None,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--final-lr", type=float, default=1.0e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--use-bfloat16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    summaries = run(build_parser().parse_args())
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
