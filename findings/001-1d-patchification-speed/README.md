# 001 — 1D Temporal Patchification Training Speed

## Summary

The p3 temporal-patch model reduces a 90-frame input to 30 transformer tokens
with `Conv1d(kernel_size=3, stride=3)`. On an NVIDIA RTX 6000 Ada, the patch
model trained **2.59x faster** than the raw-frame model at the configured batch
size of 512 while using **56.2% less peak allocated GPU memory**.

These measurements compare training-step throughput, not representation quality
or time to reach a downstream accuracy target.

## Benchmark setup

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 6000 Ada Generation, 48 GB |
| PyTorch | 2.1.0 |
| Precision | BF16 autocast |
| Encoder | Base, 12 blocks, 384 dimensions |
| Predictor | Base, 6 blocks, 192 dimensions |
| Motion input | 90 frames × 366 features |
| Raw model | `mot_base_1d` |
| Patch model | `mot_patch_base_1d`, temporal patch size 3 |
| Masks | 1 context mask, 4 target masks, overlap disabled |
| Context ratio | 0.85–1.0 |
| Target ratio | 0.15–0.2 |
| Optimizer | AdamW |
| Timed work | EMA target forward, online encoder, predictor, loss, backward, optimizer step, EMA update |
| Excluded work | Dataset loading and host-to-device input preparation |

Each model was initialized independently with the same seed. CUDA was warmed up
before timing, and every measured step was synchronized with CUDA events.

## Results

| Batch | Variant | Token frames | Context tokens | Target tokens/mask | Median step | Throughput | Peak allocated memory |
|---:|---|---:|---:|---:|---:|---:|---:|
| 128 | Raw | 90 | 24 | 16 | 27.92 ms | 4,584 samples/s | 1.89 GiB |
| 128 | Patch p3 | 30 | 9 | 5 | 21.83 ms | 5,864 samples/s | 0.97 GiB |
| 512 | Raw | 90 | 23 | 16 | 103.49 ms | 4,947 samples/s | 5.91 GiB |
| 512 | Patch p3 | 30 | 9 | 5 | 39.92 ms | 12,826 samples/s | 2.59 GiB |

At batch 128, patchification produced a **1.28x speedup** and approximately
**48.9% lower** peak memory. At batch 512, where the GPU was utilized more fully,
the speedup increased to **2.59x**, with **56.2% lower** peak memory.

The train split contains 435,690 samples. With `drop_last=true` and batch size
512, one epoch has 850 steps. Extrapolating only the measured GPU step time:

| Variant | Compute-only time/epoch | Compute-only time/300 epochs |
|---|---:|---:|
| Raw | approximately 88 seconds | approximately 7.33 hours |
| Patch p3 | approximately 34 seconds | approximately 2.83 hours |

Actual wall-clock training will be longer because this estimate excludes data
loading, checkpointing, logging, validation, and distributed synchronization.

## Interpretation and next experiment

The attention sequence is shorter, but the measured improvement is smaller than
the idealized quadratic reduction because input projection, MLP layers, optimizer
updates, EMA updates, and kernel-launch overhead do not scale quadratically with
token count. The patch model also predicts 3-frame patch embeddings rather than
individual frame embeddings, so equal step count does not imply equal learning
difficulty.

The next comparison should train raw and patch models with identical data order,
optimizer settings, update count, and evaluation schedule, then compare:

1. Wall-clock time and peak memory over complete epochs.
2. JEPA training loss, treated only as an optimization diagnostic because the
   target granularity differs.
3. 100STYLE frozen linear-probe accuracy at matched updates and matched wall time.
4. Throughput under DDP to determine whether input loading or synchronization
   reduces the single-GPU speedup.

Raw measurements are available in [benchmark-results.csv](benchmark-results.csv).
