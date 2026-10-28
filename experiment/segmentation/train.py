"""Train a frame classifier from raw motion with masked BABEL supervision."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
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
from utils.schedulers import WarmupCosineSchedule
from model.specs import MODEL_SPECS
from .metrics import FrameMetricAccumulator
from .model import MotionSegmentationProbe, MotionSegmentationTransformer, MotionSegmentationPatchTransformerProbe, MotionSegmentationMLPProbe
from .subset import select_training_subset


def shuffled_training_batches(size, batch_size, generator):
    """Repeat shuffled passes, dropping each pass's incomplete final batch."""
    if size < batch_size or batch_size < 1:
        raise ValueError("Training requires at least one complete batch")
    while True:
        order = torch.randperm(size, generator=generator).tolist()
        for start in range(0, size - batch_size + 1, batch_size):
            yield order[start:start + batch_size]


def _autocast(device, enabled):
    return torch.autocast("cuda", dtype=torch.bfloat16) if enabled and device.type == "cuda" else nullcontext()


@torch.inference_mode()
def evaluate_split(model, loader, device, use_bfloat16=False):
    model.eval()
    metrics = FrameMetricAccumulator(
        loader.dataset.num_classes, loader.dataset.class_indices_60,
    )
    loss_sum, frame_count = 0.0, 0
    for motion, fps, lengths, labels, supervised, _ in loader:
        motion, fps = motion.to(device), fps.to(device)
        labels, supervised = labels.to(device, dtype=torch.float32), supervised.to(device)
        valid = torch.arange(labels.shape[1], device=device)[None] < lengths.to(device)[:, None]
        selected = supervised.bool() & valid
        count = int(selected.sum())
        if count == 0:
            continue
        with _autocast(device, use_bfloat16):
            logits = model(motion, valid, fps)
        logits = logits.float()
        loss = F.binary_cross_entropy_with_logits(logits[selected], labels[selected])
        if not torch.isfinite(loss):
            raise ValueError("Non-finite segmentation loss")
        loss_sum += float(loss) * count
        frame_count += count
        metrics.update(logits, labels, selected)
    if not frame_count:
        raise ValueError("Split has no supervised frames")
    result = {"loss": loss_sum / frame_count, "supervised_frames": frame_count}
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


def _validate_motion_depth(args):
    from .probe import selected_feature_layers
    layers = selected_feature_layers(args)
    architecture = getattr(args, "probe_architecture", "attentive")
    if getattr(args, "feature_layers", None) is not None and (
            args.mode != "lora" and (args.mode != "probe" or args.feature_source != "jepa" or architecture not in {"patch-transformer", "mlp"})):
        raise ValueError("--feature-layers requires JEPA --mode probe --probe-architecture patch-transformer")
    if args.mode != "lora" and architecture in {"patch-transformer", "mlp"} and (
            args.mode not in ("probe", "lora") or args.feature_source != "jepa" or args.checkpoint is None or layers is None):
        raise ValueError("patch-transformer requires JEPA probe mode, checkpoint, and selected layers")
    if architecture == "mlp" and args.mode != "probe":
        raise ValueError("MLP requires frozen JEPA probe mode")
    if args.mode == "lora":
        if args.feature_source != "jepa" or args.checkpoint is None:
            raise ValueError("LoRA requires JEPA checkpoint and selected layers")
        if args.lora_rank <= 0 or (args.lora_alpha is not None and
                (not math.isfinite(args.lora_alpha) or args.lora_alpha <= 0)):
            raise ValueError("LoRA rank and alpha must be positive and finite")
        if args.precompute_batch_size <= 0:
            raise ValueError("precompute_batch_size must be positive")
    if getattr(args, "patchify", False) and args.mode != "supervised":
        raise ValueError("--patchify requires --mode supervised")
    depth = getattr(args, "motion_depth", None)
    if depth is not None:
        if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
            raise ValueError("motion_depth must be a positive integer")
        if args.mode != "supervised" and not (args.mode == "probe" and architecture == "mlp"):
            raise ValueError("--motion-depth requires supervised mode or an MLP probe")


def run(args):
    _validate_motion_depth(args)
    from experiment.representation import select_defaults
    select_defaults(args, "segmentation")
    fraction = getattr(args, "train_fraction", 1.0)
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("train_fraction must be finite and in (0, 1]")
    if args.iterations <= 0 or not 0 <= args.warmup_iterations < args.iterations:
        raise ValueError("Require iterations > 0 and 0 <= warmup_iterations < iterations")
    if min(args.eval_interval, args.checkpoint_interval, args.log_freq) <= 0:
        raise ValueError("Evaluation, checkpoint, and logging intervals must be positive")
    for name in ("start_lr", "lr", "final_lr"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if args.lr == 0 or max(args.start_lr, args.final_lr) > args.lr:
        raise ValueError("Require positive lr with start_lr and final_lr no greater than lr")
    if args.batch_size <= 0 or args.eval_batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Require positive batch sizes and num_workers >= 0")
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
    if getattr(args, "representation", None) is not None:
        from experiment.representation import validate_dataset
        validate_dataset(root, args.representation)
    datasets = {
        split: BabelSegmentationDataset(
            root, split, num_frames=meta["num_frames"], fps=meta["fps"],
            motion_dim=meta["motion_dim"], stats_root=root / "stats",
            representation=meta.get("representation", "MotionJEPAMotionRep"),
            normalize=args.mode == "supervised",
        )
        for split in ("train", "val", "test")
    }
    if not len(datasets["train"]) or not len(datasets["val"]):
        raise ValueError("Supervised segmentation requires nonempty train and validation splits")
    run_name = args.model_size if args.mode == "supervised" else (
        f"{args.model_size}-probe-{args.feature_source}"
        + (f"-layer-{args.feature_layer}" if args.feature_layer is not None else "")
    )
    if args.mode != "lora" and getattr(args, "probe_architecture", "attentive") in {"patch-transformer", "mlp"}:
        from .probe import selected_feature_layers
        run_name += "-" + args.probe_architecture + "-layers-" + "-".join(map(str, selected_feature_layers(args)))
    if getattr(args, "motion_depth", None) is not None:
        run_name += f"-depth-{args.motion_depth}"
    if getattr(args, "patchify", False):
        run_name += "-patch"
    if args.mode == "lora":
        run_name += f"-lora-r{args.lora_rank}"
    default_output = PROJECT_ROOT / "output/segmentation" / run_name
    if getattr(args, "representation", None) == "LocalMotionJEPARep":
        default_output /= "local-motion-jepa"
    if fraction != 1:
        default_output /= f"frac{fraction:g}"
    output = (args.output_root or default_output / f"seed-{args.seed}").resolve()
    findings = args.findings_root.expanduser().resolve() if args.findings_root is not None else None
    latest, best = output / "latest.pth.tar", output / "best.pth.tar"
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Output exists: {output}; use --resume or a new --output-root")
    if args.resume and not latest.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {latest}")
    datasets["train"], subset_metadata = select_training_subset(datasets["train"], fraction, args.seed)
    if len(datasets["train"]) < args.batch_size:
        raise ValueError("Training subset is smaller than batch_size; choose a common batch size for all fractions")
    if args.resume:
        saved = _torch_load_checkpoint(latest)
        saved_subset = saved["signature"].get("train_subset")
        if saved_subset != subset_metadata:
            raise ValueError("Resume training subset differs from the saved run")
    print(f"Segmentation samples: train={len(datasets['train'])} (fraction={fraction:g}), "
          f"val={len(datasets['val'])}, test={len(datasets['test'])}", flush=True)
    feature_metadata = None
    if args.mode == "lora":
        from .lora import prepare_lora_model
        model = prepare_lora_model(datasets["train"], args, device,
                                   saved["model"] if args.resume else None)
        feature_metadata = model.feature_metadata
    elif args.mode == "probe":
        from .probe import prepare_probe_datasets
        datasets, mean, std, feature_metadata = prepare_probe_datasets(datasets, args, device)
        _seed_all(args.seed)
        patch_probe = getattr(args, "probe_architecture", "attentive") in {"patch-transformer", "mlp"}
        probe_class = MotionSegmentationMLPProbe if args.probe_architecture == "mlp" else MotionSegmentationPatchTransformerProbe if patch_probe else MotionSegmentationProbe
        model = probe_class(
            feature_metadata["feature_dim"], datasets["train"].num_classes,
            meta["num_frames"], mean, std, **({"model_size": args.model_size, "motion_depth": args.motion_depth} if args.probe_architecture == "mlp" else {"model_size": args.model_size} if patch_probe else {"num_heads": args.probe_heads}),
        ).to(device)
    else:
        if args.checkpoint is not None or args.feature_layer is not None:
            raise ValueError("Checkpoint/layer arguments require --mode probe")
        model = MotionSegmentationTransformer(
            args.model_size, input_dim=meta["motion_dim"],
            num_classes=datasets["train"].num_classes, num_frames=meta["num_frames"],
            motion_depth=getattr(args, "motion_depth", None), patchify=getattr(args, "patchify", False),
        ).to(device)
    signature = {
        "architecture": model.config, "dataset_root": str(root),
        "data_hashes": {name: _sha256_file(root / name) for name in (
            "meta.json", "index.json", "train.txt", "val.txt", "test.txt", "stats/mean.npy", "stats/std.npy",
        )},
        "code_hashes": {str(path.relative_to(PROJECT_ROOT)): _sha256_file(path) for path in (
            Path(__file__), Path(__file__).with_name("model.py"),
            PROJECT_ROOT / "model/motion_transformer_1d.py",
            PROJECT_ROOT / "model/modules.py", PROJECT_ROOT / "model/pos_embs.py",
            PROJECT_ROOT / "utils/schedulers.py",
        )},
        "optimization": {name: getattr(args, name) for name in (
            "seed", "iterations", "warmup_iterations", "eval_interval", "checkpoint_interval", "log_freq",
            "batch_size", "eval_batch_size", "num_workers", "start_lr", "lr", "final_lr",
            "weight_decay", "gradient_clip", "use_bfloat16",
        )},
        "objective": "supervised_frame_bce",
        "normalization": "dataset_training_statistics",
        "selection": "validation_babel120_frame_map",
    }
    if subset_metadata is not None:
        signature["train_subset"] = subset_metadata
        for path in (Path(__file__).with_name("subset.py"), PROJECT_ROOT / "dataset/babel_segmentation.py"):
            signature["code_hashes"][str(path.relative_to(PROJECT_ROOT))] = _sha256_file(path)
    if feature_metadata is not None:
        signature["features"] = feature_metadata
        signature["normalization"] = "training_feature_statistics_excluding_padding"
        for path in (Path(__file__).with_name("probe.py"), PROJECT_ROOT / "experiment/probing.py"):
            signature["code_hashes"][str(path.relative_to(PROJECT_ROOT))] = _sha256_file(path)
    if args.mode == "lora":
        signature["normalization"] = "pretraining_input_statistics_only"
        path = Path(__file__).with_name("lora.py")
        signature["code_hashes"][str(path.relative_to(PROJECT_ROOT))] = _sha256_file(path)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineSchedule(
        optimizer, warmup_steps=args.warmup_iterations, start_lr=args.start_lr,
        ref_lr=args.lr, final_lr=args.final_lr, T_max=args.iterations,
    )
    generator = torch.Generator().manual_seed(args.seed)
    workers = 0 if args.mode == "probe" else args.num_workers
    loader = DataLoader(
        datasets["train"],
        batch_sampler=shuffled_training_batches(len(datasets["train"]), args.batch_size, generator),
        num_workers=workers, generator=generator, persistent_workers=workers > 0,
        pin_memory=device.type == "cuda",
    )
    loaders = {
        split: DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False,
                          num_workers=0 if args.mode == "probe" else args.num_workers,
                          pin_memory=device.type == "cuda")
        for split, dataset in datasets.items() if split != "train"
    }
    history, best_step, best_map = [], 0, -math.inf
    global_step = 0
    loss_sum, frame_count = 0.0, 0
    if args.resume:
        checkpoint = _torch_load_checkpoint(latest)
        if checkpoint["signature"] != signature:
            raise ValueError("Resume dataset, architecture, code, or training settings differ")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history, global_step = checkpoint["history"], checkpoint["global_step"]
        best_step, best_map = checkpoint["best_step"], checkpoint["best_map"]
        loss_sum, frame_count = checkpoint["train_accumulator"]
        restore_rng(checkpoint["rng"], generator)
        if best_step and not best.is_file():
            raise FileNotFoundError(f"Resume requires the selected checkpoint: {best}")
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json_save(signature, output / "config.json")
    if subset_metadata is not None:
        _atomic_json_save(subset_metadata, output / "train-subset.json")
    _atomic_json_save(history, output / "metrics.json")
    batches = iter(loader)
    for global_step in range(global_step + 1, args.iterations + 1):
        batch = next(batches)
        motion, fps, lengths, labels, supervised, _ = batch
        model.train()
        motion, fps = motion.to(device), fps.to(device)
        labels, supervised = labels.to(device, dtype=torch.float32), supervised.to(device)
        valid = torch.arange(labels.shape[1], device=device)[None] < lengths.to(device)[:, None]
        selected = supervised.bool() & valid
        count = int(selected.sum())
        if not count:
            raise ValueError("Training batch has no supervised frames; cannot perform an optimizer iteration")
        learning_rate = scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, args.use_bfloat16):
            logits = model(motion, valid, fps)
        loss = F.binary_cross_entropy_with_logits(logits[selected].float(), labels[selected])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite segmentation loss at iteration {global_step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        loss_sum += float(loss.detach()) * count
        frame_count += count
        if global_step % args.log_freq == 0 or global_step == args.iterations:
            print(f"Segmentation iteration={global_step}/{args.iterations} "
                  f"loss={loss_sum / frame_count:.6f} lr={learning_rate:.3e}", flush=True)
        improved = False
        if global_step % args.eval_interval == 0 or global_step == args.iterations:
            val = evaluate_split(model, loaders["val"], device, args.use_bfloat16)
            score = val["babel-120"]["frame_map"]
            if score is None:
                raise ValueError("Validation has no positive classes")
            improved = score > best_map
            if improved:
                best_map, best_step = score, global_step
            history.append({"iteration": global_step, "lr": learning_rate,
                            "train": {"loss": loss_sum / frame_count, "supervised_frames": frame_count},
                            "val": val})
            loss_sum, frame_count = 0.0, 0
            _atomic_json_save(history, output / "metrics.json")
            print(f"Segmentation validation iteration={global_step} val_frame_map={score:.6f}", flush=True)
        if improved or global_step % args.checkpoint_interval == 0 or global_step == args.iterations:
            checkpoint = {
                "signature": signature, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "best_step": best_step, "best_map": best_map, "history": history,
                "train_accumulator": (loss_sum, frame_count), "rng": capture_rng(generator),
            }
            if improved:
                _atomic_torch_save(checkpoint, best)
            _atomic_torch_save(checkpoint, latest)
    model.load_state_dict(_torch_load_checkpoint(best)["model"], strict=True)
    test = evaluate_split(model, loaders["test"], device, args.use_bfloat16) if len(datasets["test"]) else None
    summary = {
        "input_source": args.feature_source if args.mode in ("probe", "lora") else "raw",
        "mode": args.mode, "feature_layer": args.feature_layer,
        "iterations_completed": global_step, "global_step": global_step, "train_fraction": fraction,
        "samples_seen": global_step * args.batch_size,
        "equivalent_epochs": global_step * args.batch_size / len(datasets["train"]),
        "split_counts": {split: len(dataset) for split, dataset in datasets.items()},
        "best_step": best_step, "best_val_frame_map": best_map, "test": test,
        "parameters": model.parameter_counts(), "checkpoint": str(best),
        "metrics": str(output / "metrics.json"), "signature": signature,
    }
    if args.mode != "lora" and getattr(args, "probe_architecture", "attentive") in {"patch-transformer", "mlp"}:
        from .probe import selected_feature_layers
        summary["feature_layers"] = selected_feature_layers(args)
    _atomic_json_save(summary, output / "summary.json")
    if findings is not None:
        findings.mkdir(parents=True, exist_ok=True)
        _atomic_json_save(summary, findings / "results.json")
        (findings / "README.md").write_text(
            f"# Segmentation {args.mode}\n\n"
            f"Trained {args.model_size} in {args.mode} mode for {global_step} iterations on "
            f"{len(datasets['train'])} training windows (fraction {fraction:g}, seed {args.seed}) "
            f"with {signature['normalization']} and masked frame BCE. "
            f"Validation BABEL-120 frame mAP selects iteration {best_step} "
            f"(mAP {best_map:.6f}).\n\n"
            f"Checkpoint: {best}\n\nMetrics: {output / 'metrics.json'}\n\n"
            f"Test metrics: {json.dumps(test)}\n",
        )
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    from experiment.representation import add_argument
    add_argument(parser)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "dataset/babel-segmentation-120-processed-nframes30-150")
    parser.add_argument("--model-size", choices=tuple(MODEL_SPECS), default="base")
    parser.add_argument("--patchify", action="store_true", help="Three-frame raw motion patches with per-frame segmentation logits")
    parser.add_argument("--motion-depth", type=int, help="Override supervised motion blocks or MLP hidden Linear+GELU stages (MLP default: 2); retain preset width")
    parser.add_argument("--mode", choices=("supervised", "probe", "lora"), default="supervised")
    parser.add_argument("--feature-source", choices=("raw", "jepa"), default="jepa")
    feature_selection = parser.add_mutually_exclusive_group()
    feature_selection.add_argument("--feature-layer", type=int, help="One-based JEPA block output (legacy scalar)")
    feature_selection.add_argument("--feature-layers", type=int, nargs="+",
                                   help="Ordered JEPA layers to concatenate, e.g. 3 6 9 12")
    parser.add_argument("--probe-architecture", choices=("attentive", "patch-transformer", "mlp"),
                        default="attentive", help="Attentive frame head, patch Transformer, or token-wise MLP on patch features")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, help="LoRA scaling numerator; defaults to rank")
    parser.add_argument("--encoder-key", choices=("encoder", "target_encoder"), default="target_encoder")
    parser.add_argument("--precompute-batch-size", type=int, default=256)
    parser.add_argument("--probe-heads", type=int, default=6)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--findings-root", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-fraction", type=float, default=1.0,
        help="Fixed fraction of training windows with all-label coverage; uses --seed. Val/test stay unchanged.",
    )
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--warmup-iterations", type=int, default=1500)
    parser.add_argument("--eval-interval", type=int, default=3000)
    parser.add_argument("--checkpoint-interval", type=int, default=3000)
    parser.add_argument("--log-freq", type=int, default=30)
    parser.add_argument("--start-lr", type=float, default=0.)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--final-lr", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--use-bfloat16", action=argparse.BooleanOptionalAction, default=True,
                        help="CUDA BF16 mixed precision (default); disable with --no-use-bfloat16")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--describe", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    from experiment.representation import select_defaults
    select_defaults(args, "segmentation")
    _validate_motion_depth(args)
    if args.describe:
        from motion_rep.registry import get_representation
        meta_path = args.dataset_root / "meta.json"
        if meta_path.is_file():
            from experiment.representation import validate_dataset
            rep = validate_dataset(args.dataset_root, args.representation)
            meta = json.loads(meta_path.read_text())
        else:
            rep = get_representation(args.representation or "MotionJEPAMotionRep")
            meta = {"num_frames": 150}
        if args.mode == "lora":
            from .lora import LoRASegmentation
            model = LoRASegmentation(args, meta["num_frames"], 120, args.representation)
        elif args.mode == "probe":
            feature_dim = rep.FEATURE_DIM if args.feature_source == "raw" else MODEL_SPECS[args.model_size]["embed_dim"]
            if args.probe_architecture in {"patch-transformer", "mlp"}:
                from .probe import selected_feature_layers
                feature_dim *= len(selected_feature_layers(args))
                probe_class = MotionSegmentationMLPProbe if args.probe_architecture == "mlp" else MotionSegmentationPatchTransformerProbe
                model = probe_class(
                    feature_dim, 120, meta["num_frames"], torch.zeros(feature_dim), torch.ones(feature_dim),
                    **({"model_size": args.model_size, "motion_depth": args.motion_depth} if args.probe_architecture == "mlp" else {"model_size": args.model_size}))
            else:
                model = MotionSegmentationProbe(feature_dim, 120, meta["num_frames"], torch.zeros(feature_dim),
                                               torch.ones(feature_dim), num_heads=args.probe_heads)
        else:
            model = MotionSegmentationTransformer(args.model_size, input_dim=rep.FEATURE_DIM, num_frames=meta["num_frames"], motion_depth=args.motion_depth, patchify=args.patchify)
        print(json.dumps({"architecture": model.config, "parameters": model.parameter_counts()}, indent=2))
        return
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
