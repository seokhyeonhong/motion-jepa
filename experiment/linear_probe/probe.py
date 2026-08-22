"""Frozen-feature linear probing for pretrained Motion-JEPA encoders."""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import os
import random
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import MODEL_FACTORIES  # noqa: E402

from .dataset import (  # noqa: E402
    StyleMotionDataset,
    build_style_datasets,
    load_style_index,
    load_style_label_index,
)


SPLITS = ("train", "val", "test")
CACHE_FORMAT_VERSION = 1
RESULT_FILENAMES = (
    "metrics.csv",
    "summary.json",
    "linear-probe-best.pth.tar",
    "class-index.json",
)


@dataclass(frozen=True)
class Metrics:
    loss: float
    top1_accuracy: float
    macro_accuracy: float
    top5_accuracy: float


def _sha256_file(path: Path) -> str:
    stat = path.stat()
    return _sha256_file_version(path, stat.st_size, stat.st_mtime_ns)


@functools.lru_cache(maxsize=None)
def _sha256_file_version(path: Path, size: int, mtime_ns: int) -> str:
    del size, mtime_ns  # They form the cache key and detect in-process file changes.
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json_save(value: Any, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
        torch.cuda.set_device(device)
    return device


def _torch_load_checkpoint(path: Path) -> dict[str, Any]:
    """Load checkpoints written with NumPy 1.x or 2.x module names."""
    aliases: dict[str, Any] = {}
    try:
        import numpy._core  # type: ignore[attr-defined]  # noqa: F401
    except ModuleNotFoundError:
        module_aliases = {
            "numpy._core": np.core,
            "numpy._core.multiarray": np.core.multiarray,
            "numpy._core.numeric": np.core.numeric,
        }
        for name, module in module_aliases.items():
            if name not in sys.modules:
                sys.modules[name] = module
                aliases[name] = module
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    finally:
        for name, module in aliases.items():
            if sys.modules.get(name) is module:
                del sys.modules[name]
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint payload must be a mapping: {path}")
    return checkpoint


def load_frozen_encoder(
    checkpoint_path: Path,
    checkpoint_key: str,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    """Reconstruct an encoder from checkpoint config and strictly load its weights."""
    checkpoint = _torch_load_checkpoint(checkpoint_path)
    if checkpoint.get("format_version") != 1:
        raise ValueError(f"Unsupported Motion-JEPA checkpoint format: {checkpoint_path}")
    if checkpoint_key not in checkpoint:
        raise KeyError(f"Checkpoint has no {checkpoint_key!r} weights: {checkpoint_path}")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Checkpoint does not contain a training config: {checkpoint_path}")
    try:
        data_config = config["data"]
        meta_config = config["meta"]
        model_name = str(meta_config["model_name"])
        factory = MODEL_FACTORIES[model_name]
        num_frames = int(data_config["num_frames"])
        motion_dim = int(data_config["motion_dim"])
        num_joints = int(data_config["num_joints"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid model config in checkpoint: {checkpoint_path}") from error
    kwargs: dict[str, Any] = {"in_chans": motion_dim, "num_frames": num_frames}
    if model_name.endswith("_2d"):
        kwargs["num_joints"] = num_joints
    encoder = factory(**kwargs)
    encoder.load_state_dict(checkpoint[checkpoint_key], strict=True)
    encoder.to(device).eval().requires_grad_(False)
    info = {
        "model_name": model_name,
        "num_frames": num_frames,
        "motion_dim": motion_dim,
        "num_joints": num_joints,
        "fps": int(data_config["fps"]),
        "feature_dim": int(encoder.embed_dim),
        "use_bfloat16": bool(meta_config.get("use_bfloat16", False)),
    }
    return encoder, config, info


def resolve_pretraining_stats(
    config: dict[str, Any], explicit_path: Path | None
) -> Path:
    """Resolve the normalization directory used during pretraining."""
    if explicit_path is not None:
        stats_root = explicit_path.expanduser().resolve()
    else:
        data_config = config["data"]
        root_path = Path(str(data_config["root_path"])).expanduser()
        if not root_path.is_absolute():
            root_path = PROJECT_ROOT / root_path
        stats_path = Path(str(data_config.get("stats_path") or "stats")).expanduser()
        stats_root = stats_path if stats_path.is_absolute() else root_path / stats_path
        stats_root = stats_root.resolve()
    missing = [name for name in ("mean.npy", "std.npy") if not (stats_root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Pretraining statistics are missing under {stats_root}: {', '.join(missing)}"
        )
    return stats_root


def pool_encoder_output(
    output: torch.Tensor,
    valid_length: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool valid 1D frame tokens or valid 2D frame-by-joint cells."""
    frames = output.shape[1]
    valid_frames = (
        torch.arange(frames, device=output.device).unsqueeze(0)
        < valid_length.to(device=output.device).unsqueeze(1)
    )
    if output.ndim == 3:
        weights = valid_frames.unsqueeze(-1).to(dtype=output.dtype)
        return (output * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    if output.ndim == 4:
        weights = valid_frames[:, :, None, None].to(dtype=output.dtype)
        numerator = (output * weights).sum(dim=(1, 2))
        denominator = weights.sum(dim=(1, 2)) * output.shape[2]
        return numerator / denominator.clamp_min(1.0)
    raise ValueError(f"Expected encoder output [B,T,D] or [B,T,J,D], got {output.shape}")


def build_cache_metadata(
    *,
    split: str,
    checkpoint_path: Path,
    checkpoint_key: str,
    dataset_root: Path,
    stats_root: Path,
    model_info: dict[str, Any],
    class_names: list[str],
) -> dict[str, Any]:
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "split": split,
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
        "pooling": "valid_token_mean",
        "class_names": class_names,
    }


def _validate_feature_cache(payload: dict[str, Any], expected: dict[str, Any]) -> None:
    if payload.get("metadata") != expected:
        raise ValueError(
            "Feature cache metadata is stale; rerun with --recompute-features"
        )
    features = payload.get("features")
    labels = payload.get("labels")
    sample_ids = payload.get("sample_ids")
    if not isinstance(features, torch.Tensor) or features.dtype != torch.float32:
        raise ValueError("Feature cache must contain a float32 feature tensor")
    if not isinstance(labels, torch.Tensor) or labels.dtype != torch.long:
        raise ValueError("Feature cache must contain an int64 label tensor")
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("Feature cache tensors have incompatible shapes")
    if features.shape[1] != int(expected["feature_dim"]):
        raise ValueError("Feature cache dimension does not match the encoder")
    if not isinstance(sample_ids, list) or len(sample_ids) != len(features):
        raise ValueError("Feature cache sample IDs do not match feature rows")
    if not torch.isfinite(features).all():
        raise ValueError("Feature cache contains non-finite values")


def extract_features(
    encoder: nn.Module,
    dataset: StyleMotionDataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_bfloat16: bool,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    feature_batches = []
    label_batches = []
    sample_ids: list[str] = []
    with torch.inference_mode():
        for motion, fps, length, labels, ids in tqdm(loader, desc="Extract features"):
            motion = motion.to(device=device, dtype=torch.float32, non_blocking=True)
            fps = fps.to(device=device, dtype=torch.float32, non_blocking=True)
            length = length.to(device=device, non_blocking=True)
            valid_frames = (
                torch.arange(motion.shape[1], device=device).unsqueeze(0)
                < length.unsqueeze(1)
            )
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda" and use_bfloat16
                else nullcontext()
            )
            with amp_context:
                encoded = encoder(motion, fps, valid_frames=valid_frames)
                pooled = pool_encoder_output(encoded, length)
            feature_batches.append(pooled.float().cpu())
            label_batches.append(labels.to(dtype=torch.long).cpu())
            sample_ids.extend(list(ids))
    if not feature_batches:
        raise ValueError("Cannot extract features from an empty split")
    return {
        "features": torch.cat(feature_batches),
        "labels": torch.cat(label_batches),
        "sample_ids": sample_ids,
    }


def load_or_extract_split(
    *,
    split: str,
    cache_path: Path,
    metadata: dict[str, Any],
    dataset: StyleMotionDataset,
    encoder: nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_bfloat16: bool,
    recompute: bool,
) -> dict[str, Any]:
    if cache_path.is_file() and not recompute:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        _validate_feature_cache(payload, metadata)
        return payload
    payload = extract_features(
        encoder,
        dataset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        use_bfloat16=use_bfloat16,
    )
    payload["metadata"] = metadata
    _validate_feature_cache(payload, metadata)
    _atomic_torch_save(payload, cache_path)
    return payload


def evaluate_classifier(
    classifier: nn.Linear,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    num_classes: int,
) -> Metrics:
    classifier.eval()
    loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size)
    loss_sum = 0.0
    total = 0
    correct = 0
    top5_correct = 0
    class_total = torch.zeros(num_classes, dtype=torch.long)
    class_correct = torch.zeros(num_classes, dtype=torch.long)
    with torch.inference_mode():
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            logits = classifier(batch_features)
            loss_sum += float(F.cross_entropy(logits, batch_labels, reduction="sum"))
            predictions = logits.argmax(dim=1)
            matches = predictions.eq(batch_labels)
            correct += int(matches.sum())
            total += len(batch_labels)
            topk = min(5, num_classes)
            top5_correct += int(
                logits.topk(topk, dim=1).indices.eq(batch_labels[:, None]).any(dim=1).sum()
            )
            cpu_labels = batch_labels.cpu()
            class_total += torch.bincount(cpu_labels, minlength=num_classes)
            class_correct += torch.bincount(
                cpu_labels[matches.cpu()], minlength=num_classes
            )
    if total == 0:
        raise ValueError("Cannot evaluate an empty feature split")
    present = class_total > 0
    macro = (class_correct[present].float() / class_total[present].float()).mean()
    return Metrics(
        loss=loss_sum / total,
        top1_accuracy=correct / total,
        macro_accuracy=float(macro),
        top5_accuracy=top5_correct / total,
    )


def train_linear_probe(
    caches: dict[str, dict[str, Any]],
    *,
    output: Path,
    checkpoint_path: Path,
    checkpoint_key: str,
    class_names: list[str],
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    seed: int,
    run_args: dict[str, Any],
) -> dict[str, Any]:
    train_features = caches["train"]["features"]
    train_labels = caches["train"]["labels"]
    feature_dim = int(train_features.shape[1])
    num_classes = len(class_names)
    _seed_all(seed)
    classifier = nn.Linear(feature_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    metrics_path = output / "metrics.csv"
    best_path = output / "linear-probe-best.pth.tar"
    best_accuracy = -1.0
    best_epoch = 0
    fields = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_top1_accuracy",
        "train_macro_accuracy",
        "train_top5_accuracy",
        "val_loss",
        "val_top1_accuracy",
        "val_macro_accuracy",
        "val_top5_accuracy",
    ]
    with metrics_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            classifier.train()
            current_lr = float(optimizer.param_groups[0]["lr"])
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(classifier(batch_features), batch_labels)
                loss.backward()
                optimizer.step()
            train_metrics = evaluate_classifier(
                classifier,
                train_features,
                train_labels,
                device=device,
                batch_size=batch_size,
                num_classes=num_classes,
            )
            val_metrics = evaluate_classifier(
                classifier,
                caches["val"]["features"],
                caches["val"]["labels"],
                device=device,
                batch_size=batch_size,
                num_classes=num_classes,
            )
            row = {
                "epoch": epoch,
                "learning_rate": current_lr,
                **{f"train_{key}": value for key, value in asdict(train_metrics).items()},
                **{f"val_{key}": value for key, value in asdict(val_metrics).items()},
            }
            writer.writerow(row)
            file.flush()
            if val_metrics.top1_accuracy > best_accuracy:
                best_accuracy = val_metrics.top1_accuracy
                best_epoch = epoch
                _atomic_torch_save(
                    {
                        "format_version": 1,
                        "classifier": classifier.state_dict(),
                        "feature_dim": feature_dim,
                        "num_classes": num_classes,
                        "class_names": class_names,
                        "epoch": epoch,
                        "val_metrics": asdict(val_metrics),
                        "checkpoint": str(checkpoint_path),
                        "checkpoint_key": checkpoint_key,
                        "run_args": run_args,
                    },
                    best_path,
                )
            scheduler.step()

    best = torch.load(best_path, map_location=device, weights_only=False)
    classifier.load_state_dict(best["classifier"], strict=True)
    test_metrics = evaluate_classifier(
        classifier,
        caches["test"]["features"],
        caches["test"]["labels"],
        device=device,
        batch_size=batch_size,
        num_classes=num_classes,
    )
    return {
        "best_epoch": best_epoch,
        "best_val": best["val_metrics"],
        "test": asdict(test_metrics),
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "split_counts": {
            split: int(len(caches[split]["labels"])) for split in SPLITS
        },
    }


def _serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs <= 0 or args.batch_size <= 0 or args.feature_batch_size <= 0:
        raise ValueError("Epoch and batch-size arguments must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output = (
        checkpoint_path.parent / "linear-probe"
        if args.output is None
        else Path(args.output).expanduser().resolve()
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    existing_results = [name for name in RESULT_FILENAMES if (output / name).exists()]
    if existing_results and not args.overwrite:
        raise FileExistsError(
            f"Linear-probe results already exist under {output}: {existing_results}; "
            "use --overwrite to replace result files"
        )
    output.mkdir(parents=True, exist_ok=True)
    cache_root = output / "features"
    cache_root.mkdir(exist_ok=True)
    device = resolve_device(args.device)
    _seed_all(args.seed)
    encoder, config, model_info = load_frozen_encoder(
        checkpoint_path,
        args.checkpoint_key,
        device,
    )
    stats_root = resolve_pretraining_stats(config, args.stats_path)
    label_index = load_style_label_index(dataset_root)
    class_names = list(label_index.class_names)
    _atomic_json_save(
        label_index.to_json(),
        output / "class-index.json",
    )

    datasets, _ = build_style_datasets(
        dataset_root,
        splits=SPLITS,
        num_frames=model_info["num_frames"],
        fps=model_info["fps"],
        motion_dim=model_info["motion_dim"],
        stats_root=stats_root,
        label_index=label_index,
    )
    train_classes = set(datasets["train"].labels)
    expected_classes = set(range(len(class_names)))
    if train_classes != expected_classes:
        missing = [class_names[index] for index in sorted(expected_classes - train_classes)]
        raise ValueError(f"Training split does not contain every style class: {missing}")

    caches = {}
    for split in SPLITS:
        metadata = build_cache_metadata(
            split=split,
            checkpoint_path=checkpoint_path,
            checkpoint_key=args.checkpoint_key,
            dataset_root=dataset_root,
            stats_root=stats_root,
            model_info=model_info,
            class_names=class_names,
        )
        caches[split] = load_or_extract_split(
            split=split,
            cache_path=cache_root / f"{split}.pt",
            metadata=metadata,
            dataset=datasets[split],
            encoder=encoder,
            device=device,
            batch_size=args.feature_batch_size,
            num_workers=args.num_workers,
            use_bfloat16=model_info["use_bfloat16"],
            recompute=args.recompute_features,
        )

    if any(parameter.grad is not None for parameter in encoder.parameters()):
        raise RuntimeError("Frozen encoder unexpectedly accumulated gradients")
    summary = train_linear_probe(
        caches,
        output=output,
        checkpoint_path=checkpoint_path,
        checkpoint_key=args.checkpoint_key,
        class_names=class_names,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        seed=args.seed,
        run_args=_serializable_args(args),
    )
    summary.update(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_key": args.checkpoint_key,
            "dataset_root": str(dataset_root),
            "stats_root": str(stats_root),
            "model_name": model_info["model_name"],
            "pooling": "valid_token_mean",
            "seed": args.seed,
        }
    )
    _atomic_json_save(summary, output / "summary.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Linear-probe Motion-JEPA features")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "dataset/100style-processed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Linear-probe result directory "
            "(default: <checkpoint directory>/linear-probe)"
        ),
    )
    parser.add_argument(
        "--checkpoint-key",
        choices=("target_encoder", "encoder"),
        default="target_encoder",
    )
    parser.add_argument("--stats-path", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recompute-features", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
