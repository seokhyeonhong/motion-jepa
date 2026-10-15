# MotionJEPA

MotionJEPA learns motion representations by predicting target embeddings from
masked context embeddings. The repository contains a self-contained BONES-SEED
preprocessing pipeline, the `motion_jepa_366_v1` SOMA30 representation, two
MotionJEPA transformer variants, distributed pretraining, and visualization.

The code does not require the Ardy or Kimodo repositories at runtime.

## Layout

```text
dataset/          preprocessing, fixed-clip dataset, and loader
model/            frame-token and skeletal-temporal transformers
mask/             deterministic structured mask collators
motion_rep/       366-D representation and geometry
skeleton/         BVH parsing, SOMA skeletons, kinematics, and assets
visualization/    viser dataset viewer, skeleton renderer, and skinning
utils/            distributed, logging, scheduler, and tensor helpers
configs/          active pretraining config and `_depr_experiments/` archive
train.py          shared JEPA training loop
main.py           local-device and torchrun entry point
```

Public imports are intentionally top-level:

```python
from dataset import MotionDataset, make_motion_dataset
from mask import MaskCollator1D
from model import MotionTransformer1D
from motion_rep import MotionJEPAMotionRep
from skeleton import SOMASkeleton30, SOMASkeleton77, parse_bvh_motion
from visualization import MotionJEPADatasetViewer, SOMASkin
```

## Installation

Python 3.10 or newer and PyTorch 2.1 or newer are recommended.

```bash
pip install -r requirements-motion.txt
```

## Dataset preprocessing

The default preprocessing command reads BONES-SEED SOMA Uniform BVH files and
writes independently canonicalized clips as individual NumPy arrays:

```bash
python dataset/preprocess_dataset.py --workers 64
```

For a quick end-to-end check, limit preprocessing to a few source motions per
split:

```bash
python dataset/preprocess_dataset.py \
  --limit 3 \
  --workers 1 \
  --output dataset/bones-seed-processed-preview
```

The default output is `dataset/bones-seed-processed`:

```text
bones-seed-processed/
├── motions/{train,val,test}/*.npy
├── motions/{train,val,test}.json
├── train.txt
├── val.txt
├── test.txt
├── index.json
├── meta.json
├── errors.jsonl
└── stats/{mean,std}.npy
```

Each motion file stores one raw `float32[length,366]` array. Split rows contain
`sample_id,relative_npy_path,fps,actual_length`. Higher source rates are reduced only by exact
fixed-step indexing. Lower or non-divisible rates are reported in
`errors.jsonl`. Complete windows use 50% overlap by
default. After the last complete window, one uncovered tail is retained when
it has at least `--min_frames` frames (90 by default).

The final output directory is created immediately, and each sequence NPY is
saved there as soon as its source motion is converted. Split manifests,
statistics, and metadata are finalized after conversion. Complete output is
reused unless `--overwrite` is supplied; interrupted partial output requires
`--overwrite` to restart.

The loader validates `meta.json`, manifests, every split row, value size, FPS, feature
dimension, finite values, and normalization statistics. Runtime resampling is
not supported. Variable-length tails are normalized over their real frames,
then end-padded with literal zeros; padding is excluded from masks, attention,
targets, and loss.

### 100STYLE preprocessing

The 100STYLE preprocessor reads
[`Frame_Cuts.csv`](dataset/100style-soma77/Frame_Cuts.csv), trims each original
60 FPS motion with stop-exclusive `[START:STOP]` bounds, and resamples to 30 FPS.
It creates non-overlapping 150-frame windows and retains a final remainder
with at least 60 frames. Shorter remainders are discarded; a source with fewer
than 60 resampled frames contributes no windows. Each retained window is
canonicalized independently.

All windows are randomly shuffled into 85% train, 5% validation, and 10% test
using `--split_seed 42`. Train and validation counts are rounded down; test
receives the remaining windows. Splitting is by window, so a source recording
can appear in multiple splits. Split rows are shuffled reproducibly, and
normalization statistics use training frames only.

Run the default preprocessing with:

```bash
python dataset/preprocess_100style.py
```

The input is `dataset/100style-soma77` and the output is
`dataset/100style-processed-nframes60-150`. For a small preview:

```bash
python dataset/preprocess_100style.py \
  --limit 10 \
  --workers 1 \
  --output dataset/100style-processed-nframes60-150-preview
```

By default, contents are `BR/BW/FR/FW/SR/SW`; use `--contents` to select others.
`--frame-cuts` defaults to `Frame_Cuts.csv` inside `--dataset_root`.
Content names, style coverage, source availability, and frame-cut ranges are
checked strictly. Metadata records window limits, split seed, proportions,
CSV SHA256, and trim/resampling convention; incompatible output is not reused.

Workers convert and save source motions directly, avoiding transfer of large
feature arrays back to the parent process. Training uses global per-epoch
randomization through `DistributedSampler`; each loader worker lazily opens
only the NPY files selected for its current batch. NPY is the supported
processed dataset format for training and visualization.

Visualize processed clips with:

```bash
python visualize_dataset.py dataset/bones-seed-processed --split train --mesh
```

Visualize the frame-token mask timeline with:

```bash
python visualize_mask.py --config configs/mjepa_1d_base_v2.yaml --output output/mask.png --seed 0 --valid-length 90
```

The sole `MaskCollator1D` samples temporal blocks using each sample's valid
length. Blocks are padded with `-1`, and padding is excluded from attention and
loss. Configure context selection with:

```yaml
mask:
  context_selection: random  # all (default) | prefix | random
  min_context_tokens: 1
  min_context_ratio: 0.2
```

`all` retains every available context token. `prefix` takes the earliest tokens;
`random` samples without replacement and sorts the original frame indices.
The latter two trim toward the batch minimum while preserving each sample's
minimum context guard. Target blocks keep their original lengths.

There is no `mask.version` selector. Remove it from existing configs. Saved
state from the former V2 collator migrates to the unversioned name and resumes
the same masks when settings match; legacy collator state is rejected. The
`v2` suffix in dataset paths and experiment config filenames still identifies
the preprocessing dataset variant.

## Frame-token JEPA

JEPA uses one token per motion frame and temporal transformer attention.
Factories are available in `tiny`, `small`, `base`, `large`, `huge`, and `giant`
sizes, for example `mot_base_1d` and `mot_predictor_base_1d`. The 2D and temporal
patch transformers and their masking strategies have been removed.

RoPE is built into every encoder and predictor attention layer with theta 100
and a time scale of 1. Positions are physical seconds (`frame_index / fps`).
Masking gathers the original positions, preserving gaps between visible frames;
predictor context and target tokens share the same timeline. The EMA target
encoder uses the full timeline, and padding is excluded from attention and loss.
There is no positional-embedding selector or `position_encoding` configuration
section. Remove that section and any `patch` section from training configs.

```bash
python main.py --config configs/mjepa_1d_base_v2.yaml --devices cuda:0
```

Checkpoints record fixed positional semantics as architecture metadata. Existing
frame-token RoPE checkpoints with theta 100 and time scale 1 remain loadable.
Absolute-position checkpoints, checkpoints without positional metadata, custom
RoPE settings, and removed model variants fail explicitly. Archived experiment
outputs and findings remain historical records. Downstream prediction,
generation, and TMR heads retain their own positional embeddings.

## Training

Batch size is per rank. Learning rates are not automatically scaled.

Single GPU:

```bash
python main.py --config configs/mjepa_1d_base_v2.yaml --devices cuda:0
```

Local multi-GPU:

```bash
python main.py \
  --config configs/mjepa_1d_base_v2.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3
```

`main.py` spawns one process per configured device and propagates child
failures. `--debug` forces one process on the first device.

Standard torchrun launch:

```bash
torchrun --standalone --nproc-per-node=4 main.py \
  --config configs/mjepa_1d_base_v2.yaml
```

`main.py` honors `RANK`, `WORLD_SIZE`, and `LOCAL_RANK`. CUDA training uses
NCCL; CPU integration tests use Gloo. `main_distributed.py` remains available
for Submitit/SLURM launches.

Checkpoints contain unwrapped encoder and predictor weights, the EMA target,
optimizer, optional AMP scaler, all schedules, epoch/global step, mask state,
latest online diagnostics, and per-rank Python/NumPy/PyTorch RNG states. Writes are atomic. Exact resume
requires the same world size and continues from the latest completed epoch:

```yaml
meta:
  load_checkpoint: true
  read_checkpoint: null  # null selects <write_tag>-latest.pth.tar
```

The active config also evaluates a fixed unlabeled validation subset at the
linear-probe cadence. TensorBoard receives RankMe, feature variance, sample
cosine, covariance-spectrum, held-out JEPA, PredictionGain, and trajectory
reliance metrics under `online_metrics/`. Full summaries are appended to
`online-metrics.jsonl`; they are diagnostic and do not select checkpoints. See
`agent/ONLINE_REPRESENTATION_METRICS.md` for definitions and
`agent/MOTION_JEPA_TRAJECTORY.md` for the deferred cross-attention design.

BF16 autocast is controlled by `meta.use_bfloat16`. BF16 does not use a
gradient scaler; optional FP16 training uses `meta.use_float16` and restores
its scaler state.

TensorBoard logging is enabled in the standard configs. Install TensorBoard
and launch it against the training output:

```bash
pip install tensorboard
tensorboard --logdir output
```

Events are written under `<logging.folder>/tensorboard` at `logging.log_freq`.
Loss, iteration time, learning rate, and weight decay are interval averages
accumulated since the previous log event. The CSV continues to store raw values
for every optimization step.

## Linear probing

Extract and cache frozen features from a pretrained EMA target encoder, then
train a single linear style classifier on 100STYLE:

```bash
python -m experiment.linear_probe \
  --checkpoint output/<run>/motion-jepa-1d-latest.pth.tar \
  --dataset-root dataset/100style-processed-nframes60-150 \
  --output output/linear-probe/<run>
```

`--output` may be omitted; in that case results are written under
`<checkpoint directory>/linear-probe`. Supplying it explicitly selects a
different linear-probe result directory.

The encoder is reconstructed from the checkpoint config and uses the
pretraining mean and standard deviation, not the 100STYLE statistics. Its
selected pooled features are cached as `float32` under
`<output>/features`. The probe is a bias-enabled `nn.Linear` trained with
ordinary cross-entropy and momentum SGD. Use `--overwrite` to rerun the head
while retaining valid feature caches, or `--recompute-features` when the
checkpoint, dataset index, statistics, or extraction setup has changed.

Pooled-feature linear-probe commands and online probes now standardize each
feature channel using **training features only**: `(x - train_mean) /
max(train_std, 1e-6)`, with population standard deviation. Validation and test
reuse those training statistics. Raw feature caches stay unchanged; saved
offline heads include a `standardizer` containing the mean and scale required
at inference. Use `--no-standardize` in the probe/LR-sweep CLI, or
`linear_probe.standardize: false` online, to reproduce the previous protocol.
This feature transformation is additional to BONES input-motion normalization.

For the default validation-free dataset, the probe protocol is fixed in
advance: 50 epochs, SGD with LR 0.3, momentum 0.9, zero weight decay, and cosine
decay. Per-epoch training metrics are written to `metrics.csv`; the epoch-50
head is saved and `FW` test is evaluated exactly once. `summary.json` records
`selection: fixed_last_epoch` and `validation_used: false`.

To monitor representation quality during pretraining, opt a training config into
an in-memory 100STYLE probe:

```yaml
linear_probe:
  enabled: true
  # Probe cadence is independent of logging.checkpoint_freq:
  frequency: 10
  dataset_root: ./dataset/100style-processed-nframes60-150
  epochs: 50
  feature_batch_size: 256
  batch_size: 256
  num_workers: 8
  lr: 0.3
  momentum: 0.9
  weight_decay: 0.0
  seed: 42
  standardize: true
  pooling: global_mean
```

The EMA target encoder is evaluated before the first training epoch, at every
`linear_probe.frequency` epoch, and at the final epoch. When `frequency` is
omitted, it defaults to `logging.checkpoint_freq` for backward compatibility.
This cadence does not create additional named epoch checkpoints. Features
remain in memory rather than being cached per checkpoint. The fixed-protocol
`FW` metrics are diagnostic trajectories under `linear_probe/test_*`. They do
not create `<write_tag>-best-accuracy.pth.tar` and do not select a pretraining
checkpoint; use latest or a named epoch checkpoint for final comparison.
Linear probing is disabled when the section is absent or `enabled: false`.

For BABEL-60 and BABEL-120 online probing, use `linear_probe.datasets` instead
of `dataset_root`. This evaluates both datasets before training, at the configured frequency,
and at the final epoch. Each probe
fits a fresh 50-epoch SGD linear head on frozen EMA features with multi-label
BCE, selects the head by validation mAP, and records validation metrics under
`linear_probe/babel-60/*` and `linear_probe/babel-120/*`. BABEL has no labeled
test split. The best pretraining checkpoints are saved separately as
`<write_tag>-best-babel-60-map.pth.tar` and
`<write_tag>-best-babel-120-map.pth.tar`; probe results and both best scores
resume from the latest checkpoint. The existing single-dataset `dataset_root`
configuration continues to use the 100STYLE probe.

Enable an attentive probe alongside the standardized linear probe with:

```yaml
attentive_probe:
  enabled: true
  frequency: 10
  # datasets defaults to linear_probe.datasets (BABEL-60 and BABEL-120).
  epochs: 50
  feature_batch_size: 256
  batch_size: 256
  num_workers: 8
  lr: 0.0003
  weight_decay: 0.05
  warmup_epochs: 5
  final_lr: 1.0e-6
  gradient_clip: 1.0
  num_heads: 6
  seed: 42
```

The attentive head uses one learned query, one cross-attention block and a
residual MLP (ratio 4, GELU), followed by LayerNorm and classification. It trains
with AdamW (betas 0.9/0.999), warmup + cosine, and unweighted BCE. Inputs are
unpooled frozen EMA tokens; padding is excluded, and motions with no valid token
are rejected. It keeps the existing head's LayerNorm without adding the pooled
linear probe's channel standardization. Tokens are held temporarily in CPU BF16
memory, one dataset at a time; no token caches are written during pretraining.

Both probe types run sequentially on rank 0, preserving the pretraining RNG.
Attentive evaluation has its own frequency (defaulting to the linear frequency),
and also runs before training and at the final epoch. Its metrics use
`attentive_probe/babel-{60,120}/*`; its best pretraining checkpoints are
`<write_tag>-best-attentive-babel-{60,120}-map.pth.tar`. Latest checkpoints store
independent latest/best results for all four evaluations. Attentive probing can
be disabled independently or enabled with its own BABEL `datasets` mapping.

When resuming with a changed probe protocol (including raw → standardized
linear features), the affected best scores are reset and that probe evaluates
the resumed encoder before the next optimization step. Compatible probe scores
are restored. TensorBoard's `feature_standardization` scalar marks the protocol;
historical raw and standardized values should not be treated as one unchanged
evaluation curve. A running process keeps its already-loaded configuration;
these settings apply on the next launch/resume. Attentive probing adds full
head-training time at each evaluation interval.

Sweep every direct-child latest checkpoint with:

```bash
python -m experiment.linear_probe.lr_sweep --device cuda:0
```

LR sweeps and validation-selected reports reject validation-free datasets.
They are retained only for legacy datasets with a genuine non-empty validation
split, preventing `FW` test results from becoming an implicit tuning set.

Features are extracted once per checkpoint and reused across learning rates.
Completed probe runs resume from their summaries unless `--overwrite-runs` is
given.

The sweep writes its component artifacts to
`findings/000-100style-classification/linear-probe` by default.

Train matched-size supervised CNN and Transformer baselines directly on raw
100STYLE motion with:

```bash
python -m experiment.linear_probe.train_classifier \
  --model all \
  --dataset-root dataset/100style-processed-nframes60-150 \
  --seed 42 \
  --device cuda:0
```

Train the single-layer normalized raw-motion baseline with the same data
loader, optimizer, schedule, metrics, and checkpointing entry point:

```bash
python -m experiment.linear_probe.linear \
  --dataset-root dataset/100style-processed-nframes60-150 \
  --device cuda:0
```

This model takes the masked temporal mean of each normalized `[90,366]` window
and applies exactly one `Linear(366, 100)` layer. On the validation-free
content split it saves the final epoch as `classifier-final.pth.tar` and then
evaluates `FW` once. The equivalent shared-runner command is
`python -m experiment.linear_probe.train_classifier --model linear ...`.

The same classifiers can consume frozen frame-token features from a pretrained
1D MotionJEPA target encoder:

```bash
python -m experiment.linear_probe.train_classifier \
  --input-source jepa \
  --jepa-checkpoint output/<run>/motion-jepa-1d-latest.pth.tar \
  --checkpoint-key target_encoder \
  --model all \
  --dataset-root dataset/100style-processed-nframes60-150 \
  --seed 42 \
  --device cuda:0
```

Token features are cached once as BF16 under
`<checkpoint directory>/linear-probe/token-features`. Classifier outputs default
to `<checkpoint directory>/linear-probe/classifiers`; both locations can be
overridden. Classifier provenance—including the JEPA model name, checkpoint key
and SHA256, feature dimension, and normalization-statistics hashes—is recorded
once in the final summary. The best checkpoint contains only model weights and
the architecture fields required to reconstruct the classifier; resumable latest
state is removed after successful completion.

Regenerate the combined seed-42 comparison, validation-selected CSV, and plots
after both components finish:

```bash
python -m experiment.linear_probe.report
```

See the [unified 100STYLE findings](findings/000-100style-classification/README.md)
for the current CNN, CLS Transformer, and 15-probe comparison.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite covers preprocessing, representation parity, data validation,
frame-token encoders and predictors, RoPE coordinates, masking and overlap,
checkpoint restoration, launch dispatch, single-process resume, and a real
two-process Gloo DDP run. This checkout exposes one GPU, so NCCL multi-GPU
execution cannot be exercised locally.

## Attribution

MotionJEPA includes modified Apache-2.0 portions and assets derived from NVIDIA
Ardy and Kimodo. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).

The JEPA training design is based on Meta's I-JEPA implementation and paper:

```bibtex
@article{assran2023self,
  title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author={Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2301.08243},
  year={2023}
}
```

## Training objective

Training uses masked-target Smooth-L1 loss only. The predictor returns target
embeddings; visible context tokens condition those predictions. Context loss,
its warmup and distance weighting, and context-specific logging have been
removed. Remove `loss.context` from old configs. Training loss remains available
as `train/loss` in TensorBoard, `loss` in the epoch JSONL, and in the CSV log.
