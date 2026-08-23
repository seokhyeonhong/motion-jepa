# 2D Spatio-Temporal Patchification Plan

## Goal and token geometry

Keep `motion_jepa_366_v1` and the existing lossless 366-to-SOMA30 semantic
routing unchanged. Add a separate patchified 2D family whose default stem maps
`[B, 90, 30, C]` to `[B, 30, 11, D]`. Use `TokenLayout(kind="2d")` with raw
geometry `(90, 30)`, token geometry `(30, 11)`, and temporal patch size 3.

Temporal patchification uses a per-joint `Conv1d(kernel_size=3, stride=3)`.
Incomplete temporal patches are dropped and patch positions use center time
`(3i + 1) / fps`. This keeps temporal receptive fields non-overlapping.

## Hard graph patches

Use the fixed SOMA30 indices and disjoint groups below. Every raw joint belongs
to exactly one group, preventing raw target values from entering another pooled
token.

| Group | Joint indices |
|---|---|
| pelvis | 0 |
| torso | 1, 2, 3 |
| head | 4, 5, 6, 7, 8, 9 |
| left upper arm | 10, 11 |
| left lower arm/hand | 12, 13, 14, 15 |
| right upper arm | 16, 17 |
| right lower arm/hand | 18, 19, 20, 21 |
| left leg | 22, 23 |
| left foot | 24, 25 |
| right leg | 26, 27 |
| right foot | 28, 29 |

After temporal convolution, apply a graph-convolution block independently
inside each group, using only parent-child edges whose endpoints are in that
group. Hard mean-pool the group nodes to one token. Do not use cross-group graph
edges or soft/DiffPool assignments in the patch stem; global body-part exchange
is handled by the subsequent spatial transformer attention.

## Model, masking, and training

- Add `model/motion_patch_transformer_2d.py` with encoder/predictor factories
  `mot_patch_{size}_2d` and `mot_predictor_patch_{size}_2d`.
- Add `PatchMaskCollator2D` operating on `[T'=30, J'=11]`; map raw valid lengths
  with `floor(length / 3)`. Context and target rectangles are expressed only in
  the pooled grid.
- Add patch-center temporal positions and 11 learned group embeddings. The EMA
  target uses the identical patch stem and predicts pooled patch embeddings.
- Build encoder, predictor, collator, checkpoint resume, frozen probes, and token
  caches through `TokenLayout`, never from filename suffixes.
- Default names are `configs/mjepa_patch_2d_base.yaml`, output
  `mot_patch_base_2d-p3-j11-bs.32-ep.300`, and write tag
  `motion-jepa-patch-2d-p3-j11`.

## Tests and staged experiments

Verify semantic routing remains lossless before the patch stem, output shape is
`[B,30,11,D]`, incomplete/padded patches are inactive, all 30 joints are assigned
once, and group adjacency contains no cross-group edges. Change every raw value
inside a target temporal/body group and assert visible context embeddings are
unchanged. Cover multi-mask prediction, EMA targets, strict resume, frozen probe,
and token-cache length conversion.

Run ablations in order: raw 2D baseline; temporal p3 with 30 joints; temporal p3
plus group-local graph convolution without pooling; then the full p3/j11 hard
pooling model. Compare throughput, peak memory, JEPA loss, and the existing
100STYLE linear-probe metrics under otherwise identical training settings.
