"""Frozen RoPE JEPA depth probes for TMR, with linear motion readouts.

This separate runner leaves the active online LoRA trainer and helpers unchanged.
All four frozen readouts share terminal LayerNorm and valid-frame mean pooling.
"""
from __future__ import annotations

import copy
import csv
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model.modules import initialize_transformer
from experiment.linear_probe.features import _atomic_json_save, _seed_all, _sha256_file
from .dataset import PreparedPairDataset, collate_pairs, json_digest
from .features import load_prepared_datasets
from .lora import load_motion_initialization
from .losses import symmetric_multi_positive_info_nce
from .model import AlignmentConfig, _SequenceReadout, parameter_counts
from .utils import atomic_torch_save, resolve_device
from . import train


def requested_layers(depth):
    return tuple(math.ceil(depth * fraction) for fraction in (.25, .5, .75, 1.0))


@torch.inference_mode()
def frozen_readouts(encoder, motion, valid_mask, fps, layers, *, bfloat16=False):
    """One full encoder pass; preserve original time coordinates and terminal norm."""
    encoder.eval().requires_grad_(False)
    if motion.shape[1] > encoder.num_frames or not valid_mask.any(1).all():
        raise ValueError("Invalid frame count or empty valid-frame mask")
    extra = encoder.num_frames - motion.shape[1]
    active = F.pad(valid_mask.to(motion.device, torch.bool), (0, extra), value=False)
    cleaned = F.pad(motion.masked_fill(~valid_mask.to(motion.device)[..., None], 0), (0, 0, 0, extra))
    if not torch.isfinite(cleaned).all():
        raise ValueError("Valid motion must be finite")
    saved, hooks = {}, []
    def capture(layer):
        def hook(module, inputs, output):
            saved[layer] = output
        return hook
    for layer in layers:
        hooks.append(encoder.blocks[layer - 1].register_forward_hook(capture(layer)))
    try:
        with train._autocast(motion.device, bfloat16):
            final = encoder(cleaned, fps, masks=None, valid_frames=active)
            result = {}
            for layer in layers:
                tokens = encoder.norm(saved[layer]).masked_fill(~active[..., None], 0)
                if layer == len(encoder.blocks):
                    torch.testing.assert_close(tokens, final, rtol=0, atol=0)
                result[layer] = tokens.float().sum(1) / active.sum(1, keepdim=True)
    finally:
        for hook in hooks:
            hook.remove()
    return result


class PooledBank:
    """FP32 pooled vectors exposed as one-token sequences to existing TMR helpers."""
    def __init__(self, path, records, dim):
        self.path = Path(path)
        self.key_to_row = {r["sample_id"]: i for i, r in enumerate(records)}
        self.metadata = {"feature_dim": dim}
        self.values = None
    def __getstate__(self):
        return {**self.__dict__, "values": None}
    def __getitem__(self, key):
        if self.values is None:
            self.values = np.load(self.path, mmap_mode="r", allow_pickle=False)
        vector = torch.from_numpy(np.array(self.values[self.key_to_row[key]], copy=True))
        if not torch.isfinite(vector).all():
            raise ValueError("Cached readout is non-finite")
        return vector[None, :]


class LinearMotionReadout(nn.Module):
    """Only an affine motion head; channel standardization is fixed train-only."""
    def __init__(self, dim):
        super().__init__()
        self.input_dim = dim
        self.linear = nn.Linear(dim, dim)
        self.register_buffer("feature_mean", torch.zeros(dim))
        self.register_buffer("feature_scale", torch.ones(dim))
        self.apply(initialize_transformer)
    def forward(self, tokens, valid_mask):
        active = valid_mask.to(tokens.device, torch.bool)
        if tokens.ndim != 3 or tokens.shape[-1] != self.input_dim or active.shape != tokens.shape[:2] or not active.any(1).all():
            raise ValueError("Invalid linear motion readout inputs")
        cleaned = tokens.masked_fill(~active[..., None], 0)
        if not torch.isfinite(cleaned).all():
            raise ValueError("Valid motion features must be finite")
        pooled = cleaned.float().sum(1) / active.sum(1, keepdim=True)
        output = self.linear((pooled - self.feature_mean) / self.feature_scale)
        return F.normalize(output.float(), dim=-1)


class LayerProbeAlignment(nn.Module):
    def __init__(self, text_dim, motion_dim):
        super().__init__()
        config = AlignmentConfig(text_dim=text_dim, motion_dim=motion_dim, embed_dim=motion_dim,
                                 text_depth=0, text_pooling="mean", num_heads=1)
        self.config = config
        self.text_encoder = _SequenceReadout(text_dim, config, depth=0, pooling="mean")
        self.motion_encoder = LinearMotionReadout(motion_dim)
    def encode_motion(self, tokens, mask):
        return self.motion_encoder(tokens, mask)
    def encode_text(self, tokens, mask):
        return self.text_encoder(tokens, mask)
    def forward(self, motion, motion_mask, text, text_mask):
        return self.encode_motion(motion, motion_mask), self.encode_text(text, text_mask)


def _cache_signature(datasets, provenance, identity, layers, args):
    return {
        "format_version": 1, "source": identity, "data": provenance, "layers": list(layers),
        "readout": "shared_frozen_terminal_LN_then_valid_frame_FP32_mean",
        "precision": "BF16" if args.use_bfloat16 else "FP32",
        "feature_batch_size": args.eval_batch_size,
        "split_masks": {
            split: json_digest([[r["sample_id"], r["length"], r["fps"]] for r in ds.records])
            for split, ds in datasets.items()
        },
    }


def selected_layers(args, depth):
    layers = tuple(getattr(args, "layers", None) or requested_layers(depth))
    if len(set(layers)) != len(layers) or any(layer < 1 or layer > depth for layer in layers):
        raise ValueError("Probe layers must be unique one-based encoder block indices")
    return layers


def probe_optimization(args):
    config = train.optimization_configuration(args)
    if getattr(args, "adapter_lr", None) is not None:
        raise ValueError("Linear probes have no adapters; use --motion-head-lr")
    motion_lr = getattr(args, "motion_head_lr", None)
    if motion_lr is not None:
        if not math.isfinite(motion_lr) or motion_lr <= 0:
            raise ValueError("Motion head learning rate must be finite and positive")
        config["motion_head_lr"] = motion_lr
    return config


def build_probe_optimizer(model, args):
    probe_optimization(args)
    motion_lr = getattr(args, "motion_head_lr", None)
    text_lr = getattr(args, "text_head_lr", None)
    if motion_lr is None and text_lr is None:
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.AdamW([
        {"params": model.motion_encoder.parameters(), "lr": args.lr if motion_lr is None else motion_lr,
         "name": "motion_readout"},
        {"params": model.text_encoder.parameters(), "lr": args.lr if text_lr is None else text_lr,
         "name": "text_readout"},
    ], weight_decay=args.weight_decay)


def prepare_features(datasets, provenance, encoder, identity, args, device):
    root = Path(args.feature_cache_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    layers = selected_layers(args, len(encoder.blocks))
    signature = _cache_signature(datasets, provenance, identity, layers, args)
    marker = root / "manifest.json"
    if marker.exists():
        manifest = json.loads(marker.read_text())
        # A requested subset can reuse an immutable superset manifest. Validate its
        # whole recipe, then only the selected files; cancelled layers may be absent.
        cached_layers = manifest["signature"]["layers"]
        expected = _cache_signature(datasets, provenance, identity, cached_layers, args)
        if manifest["signature"] != expected or not set(layers).issubset(cached_layers):
            raise ValueError("Layer feature cache recipe or provenance differs")
        for split, source in datasets.items():
            for layer in layers:
                item = manifest["files"][split][str(layer)]
                path = Path(item["path"])
                if _sha256_file(path) != item["sha256"]:
                    raise ValueError("Layer feature cache checksum differs")
                values = np.load(path, mmap_mode="r", allow_pickle=False)
                if values.dtype != np.float32 or list(values.shape) != [len(source), encoder.embed_dim]:
                    raise ValueError("Layer feature cache shape or dtype differs")
                if not np.isfinite(values).all():
                    raise ValueError("Non-finite cached features")
                if split == "train":
                    stats = manifest["statistics"][str(layer)]
                    np.testing.assert_array_equal(values.mean(0, dtype=np.float64).astype(np.float32), stats["mean"])
                    np.testing.assert_array_equal(
                        np.maximum(values.std(0, dtype=np.float64).astype(np.float32), 1e-6), stats["scale"])
        return manifest
    encoder.to(device).eval().requires_grad_(False)
    files = {}
    for split, source in datasets.items():
        dataset = copy.copy(source)
        dataset.sample_captions = False
        loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_pairs,
                            pin_memory=device.type == "cuda")
        arrays, temps = {}, {}
        for layer in layers:
            temporary = root / f"{split}-layer-{layer:02d}.tmp.npy"
            arrays[layer] = np.lib.format.open_memmap(
                temporary, mode="w+", dtype=np.float32, shape=(len(dataset), encoder.embed_dim))
            temps[layer] = temporary
        cursor = 0
        for batch in loader:
            frames = batch["motion_tokens"].to(device, non_blocking=True)
            active = batch["motion_mask"].to(device, non_blocking=True)
            fps = torch.full((len(frames),), provenance["provenance"]["fps"], device=device)
            result = frozen_readouts(encoder, frames, active, fps, layers, bfloat16=args.use_bfloat16)
            for layer, values in result.items():
                if not torch.isfinite(values).all():
                    raise ValueError("Non-finite frozen readout")
                arrays[layer][cursor:cursor+len(frames)] = values.cpu().numpy()
            cursor += len(frames)
        if cursor != len(dataset):
            raise ValueError("Feature extraction did not cover the split")
        files[split] = {}
        for layer in layers:
            arrays[layer].flush()
            del arrays[layer]
            path = root / f"{split}-layer-{layer:02d}.npy"
            os.replace(temps[layer], path)
            files[split][str(layer)] = {"path": str(path), "sha256": _sha256_file(path),
                                       "shape": [len(dataset), encoder.embed_dim]}
        print(f"Frozen layer features complete: {split}, {cursor} motions", flush=True)
    encoder.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    statistics = {}
    for layer in layers:
        values = np.load(files["train"][str(layer)]["path"], mmap_mode="r")
        # Only pooled training vectors enter normalization; val/test never contribute.
        mean = values.mean(0, dtype=np.float64).astype(np.float32)
        std = values.std(0, dtype=np.float64).astype(np.float32)
        statistics[str(layer)] = {"mean": mean.tolist(), "scale": np.maximum(std, 1e-6).tolist()}
    manifest = {"signature": signature, "files": files, "statistics": statistics,
                "final_layer_hook_parity": True, "encoder_frozen": True}
    _atomic_json_save(manifest, marker)
    return manifest


def probe_datasets(original, manifest, layer, dim):
    result = {}
    for split, source in original.items():
        result[split] = PreparedPairDataset(
            source.records, source.text_bank,
            motion_bank=PooledBank(manifest["files"][split][str(layer)]["path"], source.records, dim),
            sample_captions=split == "train",
        )
    return result


def load_probe_checkpoint(path, experiment="tmr_layer_linear_probe"):
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("experiment") != experiment or value.get("format_version") != 1:
        raise ValueError("Not a TMR layer linear probe checkpoint")
    return value


def fit_probe(datasets, manifest, layer, args, device, *, model_factory=None, model_config=None,
              experiment="tmr_layer_linear_probe"):
    output = Path(args.output_root).expanduser().resolve() / f"layer-{layer:02d}"
    output.mkdir(parents=True, exist_ok=True)
    dim = manifest["files"]["train"][str(layer)]["shape"][1]
    text_dim = manifest["signature"]["data"]["text_dim"]
    config = {
        "model": {"text_dim": text_dim, "motion_dim": dim, "layer": layer,
                  "motion_readout": "linear_then_L2", "text_readout": "linear_LN_mean_L2"},
        "optimization": probe_optimization(args),
    }
    if model_config is not None:
        config["model"] = model_config
    provenance = {"feature_manifest_sha256": json_digest(manifest), "layer": layer}
    latest_path, best_path = output / "latest.pth.tar", output / "best.pth.tar"
    if (output / "summary.json").exists():
        summary = json.loads((output / "summary.json").read_text())
        if summary["config"] != config or summary["provenance"] != provenance or summary["epochs_completed"] != args.epochs:
            raise ValueError("Completed layer probe recipe differs")
        latest, best = load_probe_checkpoint(latest_path, experiment), load_probe_checkpoint(best_path, experiment)
        if latest["next_epoch"] != args.epochs or best["next_epoch"] != summary["best_epoch"]:
            raise ValueError("Completed layer probe checkpoints differ")
        return summary
    _seed_all(args.seed)
    model = (model_factory or LayerProbeAlignment)(text_dim, dim).to(device)
    stats = manifest["statistics"][str(layer)]
    with torch.no_grad():
        model.motion_encoder.feature_mean.copy_(torch.tensor(stats["mean"], device=device))
        model.motion_encoder.feature_scale.copy_(torch.tensor(stats["scale"], device=device))
    optimizer = build_probe_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: train._lr_factor(epoch, epochs=args.epochs,
                   warmup_epochs=args.warmup_epochs, final_factor=args.final_lr/args.lr))
    generator = torch.Generator().manual_seed(args.seed)
    history, start_epoch, best_epoch, best_score, global_step = [], 0, 0, -math.inf, 0
    if latest_path.exists():
        saved = load_probe_checkpoint(latest_path, experiment)
        if saved["config"] != config or saved["provenance"] != provenance:
            raise ValueError("Resume config or feature provenance differs")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history = saved["history"]
        start_epoch, best_epoch, best_score, global_step = (
            saved["next_epoch"], saved["best_epoch"], saved["best_val_mean_r1"], saved["global_step"])
        if len(history) != start_epoch or not best_path.exists():
            raise ValueError("Incomplete resume checkpoint pair")
        train._restore_rng(saved["rng_state"], generator)
    elif any((output/name).exists() for name in ("config.json", "metrics.csv", "best.pth.tar")):
        raise ValueError("Incomplete output has no committed latest checkpoint")
    _atomic_json_save(config, output / "config.json")
    _atomic_json_save(provenance, output / "provenance.json")
    counts = parameter_counts(model)
    _atomic_json_save(counts, output / "parameters.json")
    local_args = copy.copy(args)
    local_args.resume = start_epoch > 0
    local_args.tensorboard_dir = None
    loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, generator=generator, collate_fn=collate_pairs,
                        pin_memory=device.type == "cuda")
    with train._tensorboard_writer(local_args, output, global_step) as writer:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            loss_sum, samples, started = 0., 0, time.monotonic()
            learning_rate = optimizer.param_groups[0]["lr"]
            for batch in loader:
                count = len(batch["caption_ids"])
                if count < 2:
                    continue
                optimizer.zero_grad(set_to_none=True)
                with train._autocast(device, args.use_bfloat16):
                    motion, text = model(*[batch[key].to(device, non_blocking=True) for key in
                                          ("motion_tokens", "motion_mask", "text_tokens", "text_mask")])
                    loss = symmetric_multi_positive_info_nce(
                        motion, text, batch["caption_ids"], batch["source_ids"], batch["start_frames"],
                        batch["end_frames"], temperature=args.temperature,
                        caption_candidate_ids=batch["caption_candidate_ids"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite linear probe loss")
                loss.backward()
                if args.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                global_step += 1
                value = float(loss.detach())
                loss_sum += value * count
                samples += count
                if writer:
                    writer.add_scalar("train/loss", value, global_step)
                    writer.add_scalar("train/learning_rate", learning_rate, global_step)
            if not samples:
                raise ValueError("No contrastive training batches")
            val = train.evaluate_split(model, datasets["val"], device,
                      batch_size=args.eval_batch_size, num_workers=args.num_workers,
                      use_bfloat16=args.use_bfloat16, chunk_size=args.chunk_size)
            history.append({"epoch": epoch+1, "lr": learning_rate, "train_loss": loss_sum/samples,
                            "train_samples": samples, **{"val_"+k:v for k,v in val.items()}})
            if len(optimizer.param_groups) == 2:
                history[-1].update(motion_head_lr=optimizer.param_groups[0]["lr"],
                                   text_head_lr=optimizer.param_groups[1]["lr"])
            improved = val["mean_r1"] > best_score
            if improved:
                best_epoch, best_score = epoch+1, val["mean_r1"]
            scheduler.step()
            saved = {
                "format_version": 1, "experiment": experiment,
                "config": config, "provenance": provenance, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "next_epoch": epoch+1, "best_epoch": best_epoch, "best_val_mean_r1": best_score,
                "global_step": global_step, "history": history,
                "rng_state": train._capture_rng(generator, device),
            }
            if writer:
                writer.add_scalar("train/epoch_loss", loss_sum/samples, global_step)
                writer.add_scalar("train/epoch_seconds", time.monotonic()-started, global_step)
                for key, value in val.items():
                    writer.add_scalar("val/"+key, value, global_step)
            train._write_metrics(history, output/"metrics.csv")
            if improved:
                atomic_torch_save(saved, best_path)
            atomic_torch_save(saved, latest_path)
            if writer:
                writer.flush()
            print(f"{experiment} layer {layer} epoch {epoch+1}/{args.epochs}: val mean R@1={val['mean_r1']:.4f}, "
                  f"{time.monotonic()-started:.1f}s", flush=True)
        selected = load_probe_checkpoint(best_path, experiment)
        if selected["config"] != config or selected["provenance"] != provenance:
            raise ValueError("Selected checkpoint belongs to another probe")
        model.load_state_dict(selected["model"], strict=True)
        test = None
        if not getattr(args, "validation_only", False):
            test = train.evaluate_split(model, datasets["test"], device,
                       batch_size=args.eval_batch_size, num_workers=args.num_workers,
                       use_bfloat16=args.use_bfloat16, chunk_size=args.chunk_size)
        if writer and test is not None:
            for key, value in test.items():
                writer.add_scalar("test/"+key, value, global_step)
        summary = {
            "layer": layer, "config": config, "provenance": provenance, "best_epoch": best_epoch,
            "best_val": {key[4:]:value for key,value in history[best_epoch-1].items() if key.startswith("val_")},
            "test": test, "epochs_completed": len(history), "parameter_counts": counts,
            "trainable_parameters": counts["total"], "selection": "mean_bidirectional_val_r1",
            "best_checkpoint": str(best_path), "latest_checkpoint": str(latest_path),
            "tensorboard_dir": str(output/"tensorboard") if args.tensorboard else None,
        }
        _atomic_json_save(summary, output/"summary.json")
        return summary


def run(args):
    if args.input_source != "raw" or args.jepa_checkpoint is None:
        raise ValueError("Layer probes require raw data and the original JEPA checkpoint")
    # Use the common optimization checks without declaring online adaptation.
    check = copy.copy(args)
    check.motion_adaptation, check.model_version = "lora", "v2"
    train._validate_args(check)
    probe_optimization(args)
    device = resolve_device(args.device)
    datasets, provenance = load_prepared_datasets(**train.data_configuration(args))
    encoder, identity = load_motion_initialization(args.jepa_checkpoint, args.checkpoint_key, provenance, args.model_size)
    layers = selected_layers(args, len(encoder.blocks))
    depth = len(encoder.blocks)
    manifest = prepare_features(datasets, provenance, encoder, identity, args, device)
    del encoder
    summaries = []
    for layer in layers:
        dim = manifest["files"]["train"][str(layer)]["shape"][1]
        prepared = probe_datasets(datasets, manifest, layer, dim)
        summaries.append(fit_probe(prepared, manifest, layer, args, device))
    result = {
        "complete": True, "source": identity,
        "depth_fractions": [100 * layer / depth for layer in layers] if getattr(args, "layers", None) else [25,50,75,100],
        "layers": layers, "results": summaries,
        "selection": "validation_only" if getattr(args, "validation_only", False) else "per-layer_validation_mean_R1_epoch_then_test_once",
    }
    output = Path(args.output_root)
    _atomic_json_save(result, output/"summary.json")
    with (output/"comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=["depth_percent","layer","trainable_parameters",
                                   "best_epoch","val_mean_r1","test_t2m_r1","test_m2t_r1","test_mean_r1"])
        writer.writeheader()
        for percent, item in zip(result["depth_fractions"],summaries):
            writer.writerow(dict(depth_percent=percent,layer=item["layer"],
                trainable_parameters=item["trainable_parameters"],best_epoch=item["best_epoch"],
                val_mean_r1=item["best_val"]["mean_r1"],test_t2m_r1=(item["test"] or {}).get("t2m_r1"),
                test_m2t_r1=(item["test"] or {}).get("m2t_r1"),test_mean_r1=(item["test"] or {}).get("mean_r1")))
    return result


def build_parser():
    parser = train.build_parser()
    parser.description = __doc__
    parser.set_defaults(model_version="v2")
    parser.add_argument("--feature-cache-root", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", help="One-based block indices; defaults to four depth fractions")
    parser.add_argument("--motion-head-lr", type=float, help="Learning rate of the affine motion readout")
    return parser


def main():
    args = build_parser().parse_args()
    if args.output_root is None:
        raise ValueError("Layer probes require an explicit --output-root")
    print(json.dumps(run(args),indent=2),flush=True)


if __name__ == "__main__":
    main()
