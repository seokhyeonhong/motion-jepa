"""Masked flow, shortcut, and representation objectives."""
import math
import torch
import torch.nn.functional as F


def masked_mean(values, valid):
    if values.ndim == 3:
        values = values.float().mean(-1)
    counts = valid.sum(1)
    if (counts == 0).any():
        raise ValueError("Loss requires at least one valid token per sample")
    return values.float().masked_fill(~valid, 0).sum(1) / counts


def alignment_loss(prediction, target, valid, kind="mse"):
    if kind == "mse":
        value = (prediction.float() - target.float()).square()
    elif kind == "cosine":
        value = 1 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)
    else:
        raise ValueError("Alignment loss must be mse or cosine")
    return masked_mean(value, valid).mean()


def interpolate(clean, noise, t):
    return (1 - t[:, None, None]) * noise + t[:, None, None] * clean


@torch.no_grad()
def shortcut_target(ema, x, t, d, fps, valid):
    half = d / 2
    v1 = ema(x, t, half, fps, valid)
    middle = (x + half[:, None, None] * v1).masked_fill(~valid[..., None], 0)
    v2 = ema(middle, t + half, half, fps, valid)
    return ((v1 + v2) / 2).masked_fill(~valid[..., None], 0)


@torch.no_grad()
def make_targets(clean, fps, valid, ema, base_steps=128, shortcut_fraction=.25, generator=None):
    if base_steps < 2 or base_steps & (base_steps - 1):
        raise ValueError("base_steps must be a power of two >= 2")
    if not 0 <= shortcut_fraction < 1:
        raise ValueError("shortcut_fraction must be in [0,1)")
    batch = len(clean)
    shortcut_count = int(batch * shortcut_fraction)
    flow_count = batch - shortcut_count
    device = clean.device
    noise = torch.randn(clean.shape, device=device, generator=generator).masked_fill(~valid[..., None], 0)
    clean = clean.masked_fill(~valid[..., None], 0)
    t = torch.randint(base_steps, (batch,), device=device, generator=generator).float() / base_steps
    d = torch.full((batch,), 1 / base_steps, device=device)
    if shortcut_count:
        levels = torch.randint(int(math.log2(base_steps)), (shortcut_count,), device=device, generator=generator)
        d[flow_count:] = 2.0 ** (-levels)
        t[flow_count:] = torch.floor(torch.rand(shortcut_count, device=device, generator=generator) / d[flow_count:]) * d[flow_count:]
    xt = interpolate(clean, noise, t)
    target = clean - noise
    if shortcut_count:
        target[flow_count:] = shortcut_target(ema, xt[flow_count:], t[flow_count:], d[flow_count:],
                                             fps[flow_count:], valid[flow_count:])
    return xt, target, t, d, flow_count


@torch.no_grad()
def sample_motion(model, noise, fps, valid, steps=32, base_steps=128):
    if steps < 1 or steps > base_steps or steps & (steps - 1):
        raise ValueError("Sampling steps must be a power of two within base_steps")
    x = noise.masked_fill(~valid[..., None], 0)
    d = torch.full((len(x),), 1 / steps, device=x.device)
    for k in range(steps):
        t = torch.full_like(d, k / steps)
        x = (x + d[:, None, None] * model(x, t, d, fps, valid)).masked_fill(~valid[..., None], 0)
    return x
