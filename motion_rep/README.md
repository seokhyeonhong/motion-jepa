# Selecting a motion representation

The original representation remains the default. Supported selectors:

| Selector | Stored format | Feature width |
| --- | --- | --- |
| `MotionJEPAMotionRep` | `motion_jepa_366_v1` | 366 |
| `LocalMotionJEPARep` | `local_motion_jepa_365_v1` | 365 |

All five preprocessors (Bones-SEED, 100STYLE, AMASS, BABEL, and BABEL
segmentation) accept `--representation`. Use a separate output directory:

```sh
python -m dataset.preprocess_bones_seed \
  --representation LocalMotionJEPARep \
  --output dataset/bones-seed-local-processed-nframes150-min60
python main.py --config configs/mjepa_1d_base_local.yaml
```

The local example config uses that dataset path and a separate training output.
To modify another JEPA config, set:

```yaml
data:
  representation: LocalMotionJEPARep
  motion_dim: 365
  root_path: ./dataset/bones-seed-local-processed-nframes150-min60
```

Other data settings are still required. When `representation` is explicit,
`motion_dim` can be omitted and is resolved automatically; a conflicting width
is rejected. Configurations without the selector retain the original default.
The resolved configuration is saved in checkpoints. Resuming with a different
representation is rejected.

Preprocessing writes the selected format and width into dataset metadata and
split manifests, and computes training statistics from the selected features.
Existing output with another representation cannot be silently reused.
The loader verifies the format, width, manifests, and statistic dimensions.
Do not reuse old normalization statistics or original-representation checkpoints
for local features; preprocess and train separately.

Supervised segmentation reads the representation from its dataset metadata.
TMR cache preparation also reads dataset metadata and includes the representation
in cache provenance; prepare a separate cache for each processed dataset.
Their training model input widths follow the dataset/cache width automatically.
Raw attentive probes require an attention head count that divides the input
width: use `--probe-heads 5` for 365D local features (the default 6 does not divide
365). Frozen JEPA probes require a checkpoint trained on matching input features.

Local features preserve root height, heading-local planar velocity and yaw
velocity, pelvis-relative joint positions, heading-relative global rotations
and joint velocities, and contacts. They discard initial XZ and yaw.
Velocity channels use forward differences per second; the final difference is
repeated. Decode fixes the initial XZ and yaw to zero. Encode valid frames before
padding. Existing datasets are never converted by a training-time selector.

## Converting an existing 366D dataset

Skip BVH parsing and forward kinematics when an original processed dataset is
already available:

```sh
python -m dataset.convert_to_local \\
  --input dataset/bones-seed-processed-nframes150-min60 \\
  --output dataset/bones-seed-local-from366-nframes150-min60 \\
  --workers 8
```

The output must be a new directory. The converter reads unnormalized float32
clips, preserves sample IDs, window boundaries, splits, annotations and foot
contacts, rotates stored body features/velocities, and computes new train-only
mean/std statistics. It uses float64 internally and writes float32 clips. Small
rounding differences from direct BVH processing are expected because the source
features were already stored as float32. Initial XZ/yaw cannot be recovered and
are not needed by the local representation.

Set the training config's `data.root_path` to the converted directory and
`data.representation: LocalMotionJEPARep` (365D). The converter does not alter
source files or old statistics. It records source metadata/index hashes in
`representation_conversion`. On failure it writes `conversion-error.txt` and
leaves no completion `meta.json`; it does not silently skip failed clips. A
failed destination cannot be reused by this command; choose another new path.

For tensor-level conversion:

```python
local = LocalMotionJEPARep(fps=30).from_global_features(original_features)
```

Input shape is `[T, 366]` or `[B, T, 366]`, with valid frames only.
