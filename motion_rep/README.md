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
