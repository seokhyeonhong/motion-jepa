"""Train a frame classifier from raw motion with masked BABEL supervision."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.babel_segmentation import BabelSegmentationDataset
from experiment.utils import (
    PROJECT_ROOT, _atomic_json_save, _atomic_torch_save, _seed_all,
    _sha256_file, _torch_load_checkpoint, resolve_device,
)
from model.specs import MODEL_SPECS
from .metrics import FrameMetricAccumulator
from .model import MotionSegmentationTransformer


def run_epoch(model, loader, device, *, optimizer=None, gradient_clip=1.0):
    training = optimizer is not None
    model.train(training)
    metrics = FrameMetricAccumulator(
        loader.dataset.num_classes, loader.dataset.class_indices_60,
    )
    loss_sum, frame_count = 0.0, 0
    for motion, fps, lengths, labels, supervised, _ in loader:
        motion, fps = motion.to(device), fps.to(device)
        labels, supervised = labels.to(device), supervised.to(device)
        valid = torch.arange(motion.shape[1], device=device)[None] < lengths.to(device)[:, None]
        selected = supervised.bool() & valid
        count = int(selected.sum())
        if count == 0:
            continue
        with torch.set_grad_enabled(training):
            logits = model(motion, valid, fps)
            loss = F.binary_cross_entropy_with_logits(logits[selected], labels[selected])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite segmentation loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
        loss_sum += float(loss.detach()) * count
        frame_count += count
        if not training:
            metrics.update(logits, labels, selected)
    if not frame_count:
        raise ValueError("Split has no supervised frames")
    result = {"loss": loss_sum / frame_count, "supervised_frames": frame_count}
    if not training:
        result.update(metrics.compute())
    return result


def capture_rng(generator):
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(), "loader": generator.get_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def run(args):
    if args.epochs <= 0 or not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("Require epochs > 0 and 0 <= warmup_epochs < epochs")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Require batch_size > 0 and num_workers >= 0")
    if not (math.isfinite(args.lr) and args.lr > 0):
        raise ValueError("lr must be finite and positive")
    if not (math.isfinite(args.weight_decay) and args.weight_decay >= 0):
        raise ValueError("weight_decay must be finite and nonnegative")
    if not (math.isfinite(args.gradient_clip) and args.gradient_clip > 0):
        raise ValueError("gradient_clip must be finite and positive")
    device = resolve_device(args.device)
    _seed_all(args.seed)
    root = args.dataset_root.expanduser().resolve()
    meta = json.loads((root / "meta.json").read_text())
    datasets = {
        split: BabelSegmentationDataset(
            root, split, num_frames=meta["num_frames"], fps=meta["fps"],
            motion_dim=meta["motion_dim"], stats_root=root / "stats",
        )
        for split in ("train", "val", "test")
    }
    if not len(datasets["train"]) or not len(datasets["val"]):
        raise ValueError("Supervised segmentation requires nonempty train and validation splits")
    model = MotionSegmentationTransformer(
        args.model_size, input_dim=meta["motion_dim"],
        num_classes=datasets["train"].num_classes, num_frames=meta["num_frames"],
    ).to(device)
    output = (args.output_root or PROJECT_ROOT / f"output/segmentation/{args.model_size}/seed-{args.seed}").resolve()
    findings = (args.findings_root or PROJECT_ROOT / f"findings/segmentation/{args.model_size}/seed-{args.seed}").resolve()
    latest, best = output / "latest.pth.tar", output / "best.pth.tar"
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Output exists: {output}; use --resume or a new --output-root")
    if args.resume and not latest.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {latest}")
    signature = {
        "architecture": model.config, "dataset_root": str(root),
        "data_hashes": {name: _sha256_file(root / name) for name in (
            "meta.json", "index.json", "train.txt", "val.txt", "test.txt", "stats/mean.npy", "stats/std.npy",
        )},
        "code_hashes": {str(path.relative_to(PROJECT_ROOT)): _sha256_file(path) for path in (
            Path(__file__), Path(__file__).with_name("model.py"),
            PROJECT_ROOT / "model/motion_transformer_1d.py",
            PROJECT_ROOT / "model/modules.py", PROJECT_ROOT / "model/pos_embs.py",
        )},
        "optimization": {name: getattr(args, name) for name in (
            "seed", "epochs", "warmup_epochs", "batch_size", "num_workers",
            "lr", "weight_decay", "gradient_clip",
        )},
        "objective": "supervised_frame_bce",
        "normalization": "dataset_training_statistics",
        "selection": "validation_babel120_frame_map",
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    def lr_factor(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(args.warmup_epochs, 1)
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs - 1, 1)
        return 0.01 + 0.99 * (1 + math.cos(math.pi * min(progress, 1.0))) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        split: DataLoader(
            dataset, batch_size=args.batch_size, shuffle=split == "train",
            num_workers=args.num_workers, generator=generator if split == "train" else None,
            pin_memory=device.type == "cuda",
        )
        for split, dataset in datasets.items()
    }
    history, start, best_epoch, best_map = [], 0, 0, -math.inf
    if args.resume:
        checkpoint = _torch_load_checkpoint(latest)
        if checkpoint["signature"] != signature:
            raise ValueError("Resume dataset, architecture, code, or training settings differ")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history, start = checkpoint["history"], checkpoint["next_epoch"]
        best_epoch, best_map = checkpoint["best_epoch"], checkpoint["best_map"]
        restore_rng(checkpoint["rng"], generator)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json_save(signature, output / "config.json")
    for epoch in range(start, args.epochs):
        train = run_epoch(model, loaders["train"], device, optimizer=optimizer, gradient_clip=args.gradient_clip)
        val = run_epoch(model, loaders["val"], device)
        score = val["babel-120"]["frame_map"]
        if score is None:
            raise ValueError("Validation has no positive classes")
        improved = score > best_map
        if improved:
            best_map, best_epoch = score, epoch + 1
        history.append({"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"], "train": train, "val": val})
        scheduler.step()
        checkpoint = {
            "signature": signature, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "next_epoch": epoch + 1,
            "best_epoch": best_epoch, "best_map": best_map, "history": history,
            "rng": capture_rng(generator),
        }
        _atomic_torch_save(checkpoint, latest)
        if improved:
            _atomic_torch_save(checkpoint, best)
        _atomic_json_save(history, output / "metrics.json")
        print(f"epoch={epoch + 1} train_loss={train['loss']:.6f} val_frame_map={score:.6f}", flush=True)
    model.load_state_dict(_torch_load_checkpoint(best)["model"], strict=True)
    test = run_epoch(model, loaders["test"], device) if len(datasets["test"]) else None
    summary = {
        "input_source": "raw", "actual_epochs": len(history),
        "best_epoch": best_epoch, "best_val_frame_map": best_map, "test": test,
        "parameters": model.parameter_counts(), "checkpoint": str(best),
        "metrics": str(output / "metrics.json"), "signature": signature,
    }
    _atomic_json_save(summary, output / "summary.json")
    findings.mkdir(parents=True, exist_ok=True)
    _atomic_json_save(summary, findings / "results.json")
    (findings / "README.md").write_text(
        f"# Raw-motion segmentation\n\n"
        f"Trained {args.model_size} from scratch for {len(history)} epochs using dataset training statistics "
        f"and masked frame BCE. Validation BABEL-120 frame mAP selects epoch {best_epoch} "
        f"(mAP {best_map:.6f}).\n\n"
        f"Checkpoint: {best}\n\nMetrics: {output / 'metrics.json'}\n\n"
        f"Test metrics: {json.dumps(test)}\n",
    )
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "dataset/babel-segmentation-120-processed-nframes30-150")
    parser.add_argument("--model-size", choices=tuple(MODEL_SPECS), default="base")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--findings-root", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--describe", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.describe:
        model = MotionSegmentationTransformer(args.model_size)
        print(json.dumps({"architecture": model.config, "parameters": model.parameter_counts()}, indent=2))
        return
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
