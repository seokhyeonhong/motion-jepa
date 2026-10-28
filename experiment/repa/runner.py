"""Training, reproducible validation, and sampling entry points."""
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from dataset.motion_dataset import MotionDataset, OffsetDistributedSampler
from experiment.utils import _atomic_torch_save, _seed_all, _torch_load_checkpoint
from motion_rep.registry import get_representation
from .model import MotionShortcutTransformer, TrainingModel
from .objectives import alignment_loss, make_targets, masked_mean, sample_motion
from .teacher import AlignmentTeacher


def setup(device_name):
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    use_cuda = device_name != "cpu" and torch.cuda.is_available()
    device = torch.device(f"cuda:{local}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if use_cuda else "gloo")
    return device, rank, world


def dataset(config, split):
    data = config["data"]
    ds = MotionDataset(root_path=data["root_path"], meta_files=data.get(f"{split}_files", [f"{split}.txt"]),
                         num_frames=data["num_frames"], fps=data["fps"], normalize=data.get("normalize", True),
                         stats_path=data.get("stats_path", "stats"), representation=data["representation"])
    if ds.normalize and (np.load(ds.stats_root / "std.npy", allow_pickle=False) < 0).any():
        raise ValueError("Normalization standard deviations must be nonnegative")
    return ds


def statistics(ds):
    return (np.zeros(ds.motion_dim, np.float32), np.ones(ds.motion_dim, np.float32)) if not ds.normalize else (ds.mean, ds.std)


def build(config, device, mean, std):
    data = config["data"]
    alignment = config["alignment"]
    shortcut = config["shortcut"]
    if not 0 <= shortcut["fraction"] < 1 or not 0 <= shortcut["ema"] < 1 or not math.isfinite(shortcut["weight"]) or shortcut["weight"] < 0:
        raise ValueError("Invalid shortcut configuration")
    base = shortcut["base_steps"]
    if not isinstance(base, int) or base < 2 or base & (base - 1):
        raise ValueError("base_steps must be a power of two >= 2")
    if not math.isfinite(alignment["weight"]) or alignment["weight"] < 0 or alignment["loss"] not in ("mse", "cosine"):
        raise ValueError("Invalid alignment weight or loss")
    teacher = AlignmentTeacher(alignment, data, mean, std, device) if alignment["weight"] else None
    motion_dim = get_representation(data["representation"]).FEATURE_DIM
    generator = MotionShortcutTransformer(in_chans=motion_dim, num_frames=data["num_frames"], **config["model"]).to(device)
    model = TrainingModel(generator, None if teacher is None else teacher.feature_dim).to(device)
    ema = copy.deepcopy(generator).requires_grad_(False).eval()
    return model, ema, teacher


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"].cpu())


def batch_to_device(batch, device, frames):
    clean, fps, lengths = [value.to(device) for value in batch]
    if ((lengths <= 0) | (lengths > frames)).any():
        raise ValueError("Invalid sequence length")
    valid = torch.arange(frames, device=device)[None] < lengths[:, None]
    return clean.float(), fps.float(), valid


def compute_loss(model, ema, teacher, batch, config, device, random_generator=None):
    clean, fps, valid = batch_to_device(batch, device, config["data"]["num_frames"])
    shortcut = config["shortcut"]
    enabled = config["training"].get("bf16", True) and device.type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=enabled):
        xt, target, t, d, nflow = make_targets(clean, fps, valid, ema, shortcut["base_steps"],
                                              shortcut["fraction"], random_generator)
        prediction, projected = model(xt, t, d, fps, valid, nflow, 1 if teacher is None else teacher.patch_size)
    per_sample = masked_mean((prediction.float() - target.float()).square(), valid)
    flow = per_sample[:nflow].mean()
    nshort = len(clean) - nflow
    consistency = per_sample[nflow:].mean() if nshort else flow.new_zeros(())
    loss = (nflow * flow + nshort * shortcut["weight"] * consistency) / len(clean)
    align = flow.new_zeros(())
    metrics = {"flow": (flow.detach() * nflow, nflow),
               "consistency": (consistency.detach() * nshort, nshort)}
    if teacher is not None:
        targets, token_valid = teacher(clean[:nflow], fps[:nflow], valid[:nflow])
        align = alignment_loss(projected, targets, token_valid, config["alignment"]["loss"])
        loss = loss + config["alignment"]["weight"] * align
        for name, values in (
            ("feature_norm_student", projected.float().norm(dim=-1)),
            ("feature_norm_teacher", targets.float().norm(dim=-1)),
            ("feature_cosine", torch.nn.functional.cosine_similarity(projected.float(), targets.float(), dim=-1))):
            metrics[name] = (masked_mean(values, token_valid).detach().sum(), nflow)
    else:
        for name in ("feature_norm_student", "feature_norm_teacher", "feature_cosine"):
            metrics[name] = (flow.new_zeros(()), 0)
    metrics["alignment"] = (align.detach() * nflow, nflow if teacher else 0)
    metrics["loss"] = (loss.detach() * len(clean), len(clean))
    for level in range(int(math.log2(shortcut["base_steps"]))):
        selected = d[nflow:] == 2. ** (-level)
        metrics[f"consistency_d_1_over_{2**level}"] = (per_sample[nflow:][selected].detach().sum(), int(selected.sum()))
    return loss, metrics


def reduce_metrics(metrics, device, world):
    keys = sorted(metrics)
    packed = torch.tensor([[float(metrics[k][0]), metrics[k][1]] for k in keys], dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(packed)
    return {key: float(row[0] / row[1]) if row[1] else 0. for key, row in zip(keys, packed.cpu())}


@torch.no_grad()
def evaluate(model, ema, teacher, ds, config, device, rank, world):
    was_training = model.training
    model.eval()
    count = min(config["validation"]["samples"], len(ds))
    if count == 0:
        raise ValueError("Validation dataset is empty")
    order = torch.randperm(len(ds), generator=torch.Generator().manual_seed(42))[:count].tolist()
    # Shard complete fixed batches: results and branch proportions do not depend on world size.
    batch_size = config["training"]["batch_size"]
    aggregate = None
    for batch_index, start in enumerate(range(0, count, batch_size)):
        if batch_index % world != rank:
            continue
        batch = torch.utils.data.default_collate([ds[i] for i in order[start:start + batch_size]])
        rng = torch.Generator(device=device).manual_seed(42000 + batch_index)
        _, metrics = compute_loss(model, ema, teacher, batch, config, device, rng)
        if aggregate is None:
            aggregate = {k: [0., 0] for k in metrics}
        for key, (value, n) in metrics.items():
            aggregate[key][0] += float(value)
            aggregate[key][1] += n
    # Every rank needs identical keys even if the subset has fewer batches than ranks.
    if aggregate is None:
        batch = torch.utils.data.default_collate([ds[order[0]]])
        _, empty = compute_loss(model, ema, teacher, batch, config, device, torch.Generator(device=device).manual_seed(42))
        aggregate = {key: [0., 0] for key in empty}
    result = reduce_metrics(aggregate, device, world)
    model.train(was_training)
    return result


def save_checkpoint(path, model, ema, optimizer, scheduler, config, step, epoch, offset, mean, std, teacher, rank, world):
    local = {"rng": rng_state(), "epoch": epoch, "offset": offset}
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, local)
    else:
        states[0] = local
    if rank == 0:
        _atomic_torch_save({"model": model.state_dict(), "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "config": config, "step": step, "rank_states": states, "world_size": world,
            "mean": mean, "std": std, "teacher": teacher.metadata if teacher else None}, path)


def train(config, output, resume=None, device_name="auto", max_steps=None):
    device, rank, world = setup(device_name)
    _seed_all(config["training"]["seed"] + rank)
    train_ds, val_ds = dataset(config, "train"), dataset(config, "val")
    mean, std = statistics(train_ds)
    val_mean, val_std = statistics(val_ds)
    if not np.array_equal(mean, val_mean) or not np.array_equal(std, val_std):
        raise ValueError("Train and validation statistics differ")
    model, ema, teacher = build(config, device, mean, std)
    opt = config["training"]
    if opt["batch_size"] < 1 or opt["iterations"] < 1 or opt["warmup"] < 0:
        raise ValueError("Invalid training schedule")
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt["lr"], betas=(.9, .999), weight_decay=opt["weight_decay"])
    def lr_factor(step):
        if opt["warmup"] and step < opt["warmup"]:
            return (step + 1) / opt["warmup"]
        progress = min(1., max(0., (step - opt["warmup"]) / max(1, opt["iterations"] - opt["warmup"])))
        ratio = opt["final_lr"] / opt["lr"]
        return ratio + (1 - ratio) * .5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    epoch = offset = step = 0
    saved_rng = None
    if resume:
        saved = _torch_load_checkpoint(Path(resume))
        if saved["config"] != config or saved["world_size"] != world:
            raise ValueError("Resume requires the same configuration and world size")
        if not np.array_equal(saved["mean"], mean) or not np.array_equal(saved["std"], std):
            raise ValueError("Resume normalization statistics changed")
        if saved["teacher"] != (teacher.metadata if teacher else None):
            raise ValueError("Resume teacher identity changed")
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        step = saved["step"]
        state = saved["rank_states"][rank]
        epoch, offset, saved_rng = state["epoch"], state["offset"], state["rng"]
    if world > 1:
        wrapped = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
        # DDP synchronizes student initialization; synchronize EMA with that student.
        if not resume:
            ema.load_state_dict(model.generator.state_dict())
    else:
        wrapped = model
    if saved_rng:
        restore_rng(saved_rng)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    writer = None
    if rank == 0:
        (output / "config.yaml").write_text(yaml.safe_dump(config))
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(output / "tensorboard"), purge_step=step if resume else None)
    sampler = OffsetDistributedSampler(train_ds, world, rank, shuffle=True, seed=opt["seed"], drop_last=True)
    loader = DataLoader(train_ds, batch_size=opt["batch_size"], sampler=sampler,
                        num_workers=opt["workers"], drop_last=True,
                        generator=torch.Generator().manual_seed(opt["seed"] + rank))
    batches_per_epoch = len(loader)
    if not batches_per_epoch:
        raise ValueError("Training dataset is smaller than one distributed batch")
    limit = opt["iterations"] if max_steps is None else min(opt["iterations"], step + max_steps)
    model.train()
    tick = time.perf_counter()
    while step < limit:
        sampler.set_epoch(epoch)
        sampler.start_index = offset * opt["batch_size"]
        for index, batch in enumerate(loader, start=offset):
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = compute_loss(wrapped, ema, teacher, batch, config, device)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), opt["clip_grad"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                for target, source in zip(ema.parameters(), model.generator.parameters()):
                    target.lerp_(source, 1 - config["shortcut"]["ema"])
            step += 1
            offset = index + 1
            if step % opt["log_interval"] == 0 or step == limit:
                values = reduce_metrics(metrics, device, world)
                values["gradient_norm"] = float(norm)
                values["lr"] = scheduler.get_last_lr()[0]
                values["samples_per_second"] = opt["batch_size"] * world / max(time.perf_counter() - tick, 1e-9)
                if rank == 0:
                    print(json.dumps({"step": step, **values}), flush=True)
                    for key, value in values.items():
                        writer.add_scalar("train/" + key, value, step)
            if step % config["validation"]["interval"] == 0 or step == limit:
                values = evaluate(model, ema, teacher, val_ds, config, device, rank, world)
                if rank == 0:
                    print(json.dumps({"step": step, "validation": values}), flush=True)
                    for key, value in values.items():
                        writer.add_scalar("validation/" + key, value, step)
            if step % opt["checkpoint_interval"] == 0 or step == limit:
                save_checkpoint(output / "latest.pt", model, ema, optimizer, scheduler, config, step,
                                epoch, offset, mean, std, teacher, rank, world)
                if rank == 0 and step % opt["checkpoint_interval"] == 0:
                    # Retain milestone checkpoints without serializing the payload again.
                    import shutil
                    shutil.copy2(output / "latest.pt", output / f"step-{step:08d}.pt")
            tick = time.perf_counter()
            if step >= limit:
                break
        if offset >= batches_per_epoch:
            epoch += 1
            offset = 0
    if writer:
        writer.close()
    if world > 1:
        dist.destroy_process_group()


def evaluate_checkpoint(checkpoint, output, device_name):
    device, rank, world = setup(device_name)
    saved = _torch_load_checkpoint(Path(checkpoint))
    config = saved["config"]
    model, ema, teacher = build(config, device, saved["mean"], saved["std"])
    model.load_state_dict(saved["model"])
    ema.load_state_dict(saved["ema"])
    if saved["teacher"] != (teacher.metadata if teacher else None):
        raise ValueError("Evaluation teacher identity changed")
    ds = dataset(config, "val")
    mean, std = statistics(ds)
    if not np.array_equal(saved["mean"], mean) or not np.array_equal(saved["std"], std):
        raise ValueError("Evaluation normalization statistics changed")
    result = evaluate(model, ema, teacher, ds, config, device, rank, world)
    if rank == 0:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
    if world > 1:
        dist.destroy_process_group()


def sample_checkpoint(checkpoint, output, steps, count, length, fps, seed, device_name):
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise ValueError("Sampling uses a single process")
    device, _, _ = setup(device_name)
    saved = _torch_load_checkpoint(Path(checkpoint))
    config = saved["config"]
    data = config["data"]
    if count < 1 or not 1 <= length <= data["num_frames"] or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid sample count, length or FPS")
    model = MotionShortcutTransformer(in_chans=len(saved["mean"]), num_frames=data["num_frames"], **config["model"]).to(device)
    model.load_state_dict(saved["ema"])
    model.eval()
    rng = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn((count, data["num_frames"], len(saved["mean"])), device=device, generator=rng)
    valid = torch.arange(data["num_frames"], device=device)[None].expand(count, -1) < length
    rates = torch.full((count,), fps, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = sample_motion(model, noise, rates, valid, steps, config["shortcut"]["base_steps"])
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result = result.float().cpu().numpy() * saved["std"] + saved["mean"]
    result[:, length:] = 0
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "motions.npy", result, allow_pickle=False)
    metadata = {"lengths": [length] * count, "fps": fps, "representation": data["representation"],
                "seed": seed, "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_step": saved["step"],
                "steps": steps, "model_evaluations": steps, "seconds": elapsed, "seconds_per_sample": elapsed / count}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata))
