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
experiment/       raw-motion classification, segmentation, and text-motion retrieval
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
feature arrays back to the parent process. Training uses global per-pass
randomization through `DistributedSampler`; each loader worker lazily opens
only the NPY files selected for its current batch. NPY is the supported
processed dataset format for training and visualization.

Visualize processed clips with:

```bash
python visualize_dataset.py dataset/bones-seed-processed --split train --mesh
```

Visualize the frame-token mask timeline with:

```bash
python visualize_mask.py --config configs/mjepa_1d_base.yaml --output output/mask.png --seed 0 --valid-length 90
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
the same masks when settings match; legacy collator state is rejected.
The default training config is `configs/mjepa_1d_base.yaml`, using the
`bones-seed-processed-nframes150-min60` dataset without version selectors.

## Frame-token JEPA

JEPA uses one token per motion frame by default and temporal transformer attention.
Factories are available in `tiny`, `small`, `base`, `large`, `huge`, and `giant`
sizes, for example `mot_base_1d` and `mot_predictor_base_1d`. Set
`meta.patchify: true` to project each group of three frames into one token with
a Conv1d (kernel size 3, stride 3, no padding). The default is `false`.
Patch mode produces `floor(num_frames / 3)` tokens, drops incomplete trailing
groups, and considers a patch valid only when all three frames are valid.
Encoder and predictor RoPE use patch-center times `(3 * token_index + 1) / fps`;
masking and online metrics operate on the resulting token grid. The previous
2D and temporal-patch model variants remain removed.

RoPE is built into every encoder and predictor attention layer with theta 100
and a time scale of 1. Positions are physical seconds (`frame_index / fps`).
Masking gathers the original positions, preserving gaps between visible frames;
predictor context and target tokens share the same timeline. The EMA target
encoder uses the full timeline, and padding is excluded from attention and loss.
There is no positional-embedding selector or `position_encoding` configuration
section. Remove that section and any `patch` section from training configs.

```bash
python main.py --config configs/mjepa_1d_base.yaml --devices cuda:0
```

Checkpoints record fixed positional semantics as architecture metadata. Existing
frame-token RoPE model weights with theta 100 and time scale 1 remain loadable.
Absolute-position checkpoints, checkpoints without positional metadata, custom
RoPE settings, and removed model variants fail explicitly. Archived experiment
outputs and findings remain historical records. Downstream prediction,
generation, and TMR heads retain their own positional embeddings.

### Optional centering and sharpening

Enable S-JEPA-style distribution matching in either pretraining config:

```yaml
loss:
  centering_sharpening:
    enabled: true
    center_momentum: 0.9
    teacher_temperature: 0.06
    student_temperature: 0.1
```

When disabled or omitted, training keeps the existing LayerNorm teacher targets
and Smooth L1 loss. When enabled, masked teacher features are centered by a
running channel mean and sharpened with softmax; the predictor uses log-softmax
without centering. Cross-entropy replaces Smooth L1, directly on the existing
feature channels, with no extra projection head or target LayerNorm. The
encoder's built-in final normalization remains part of the backbone.

This follows the target/predictor placement and temperatures in
[S-JEPA, Sections 3 and 4.2](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf).
For variable-length frame tokens, the center is one shared channel vector:
its update averages valid masked target tokens across all GPUs, before repeating
for context masks. Overlapping target blocks count separately. The loss averages
each mask/sample equally. Softmax and center arithmetic use float32. Padding
contributes to neither loss nor center. Each loss uses the previous center;
the center EMA updates once after the training step.

The center is checkpointed and restored. Resume requires the same enabled loss
and temperature/momentum settings; old reconstruction checkpoints remain
resumable with the option disabled. Start a new run/output folder to switch
objectives. Online evaluation reports `heldout_jepa/cross_entropy` using the
frozen training center, while raw representation diagnostics remain unchanged.
Smooth L1/MSE/cosine are additional feature diagnostics in this mode, rather than
the optimized objective. No negative samples are required.

## Online representation metrics

The default config evaluates a fixed, seeded subset of up to 1,024 validation
clips at iteration 0, every 1,000 iterations, and at the final iteration.
Control this with `online_metrics.enabled`, `interval`, `meta_file`,
`num_samples`, `batch_size`, `num_workers`, and `seed`. Metrics are written only
to TensorBoard under `online_metrics/`; no feature caches or CSV/JSON logs are
created. Regular `train/` metrics retain their separate 100-iteration interval.

Representation diagnostics use valid-frame mean-pooled EMA encoder features:
RankMe (uncentered singular values), mean channel standard deviation, average
pairwise cosine similarity, covariance effective rank, largest eigenvalue share,
condition number, and off-diagonal covariance magnitude. Raw and per-channel
standardized features are reported separately, along with temporal variation,
near-zero channel counts. Fixed sample/target-token counts, feature dimension,
and rank ceilings remain in the returned diagnostic summary but are excluded
from TensorBoard scalar curves. Standardized mean standard deviation is also
omitted because channel standardization fixes it near 1. Raw standard deviation
and near-zero channel count remain useful collapse diagnostics. Rank is a collapse
diagnostic, not evidence of semantic quality on its own.

Held-out JEPA diagnostics compare the online encoder/predictor against the
EMA target (LayerNorm-normalized in the default mode) using fixed independent masks: Smooth L1, MSE, cosine
similarity, and MSE gain relative to the validation targets' constant mean.
Smooth L1 weights each mask/sample equally, matching training; MSE and cosine
weight valid target tokens equally. Padding is excluded everywhere. Evaluation
runs on rank zero; other ranks wait. Training model modes, RNG state, and the
training mask stream are preserved. Obsolete spatial-token diagnostics do not
apply to the current frame-token model.

## Training

Training is controlled by optimizer iterations, independent of dataset size:

```yaml
optimization:
  iterations: 200000
  warmup_iterations: 20000
logging:
  checkpoint_interval: 10000
  log_freq: 100
```

One iteration processes one batch per rank. The loader reshuffles and starts
another data pass as necessary, stopping exactly at `iterations`, including
partway through a pass. Learning-rate, weight-decay, and EMA schedules use this
same iteration budget. Gradient clipping starts after warmup. `epochs`, `warmup`,
`ipe_scale`, and the old epoch-based `checkpoint_freq` are rejected.

Batch size is per rank. Learning rates are not automatically scaled. Changing
GPU count changes the samples processed per iteration, not the iteration budget.

Single GPU:

```bash
python main.py --config configs/mjepa_1d_base.yaml --devices cuda:0
```

Local multi-GPU:

```bash
python main.py \
  --config configs/mjepa_1d_base.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3
```

`main.py` spawns one process per configured device and propagates child
failures. `--debug` forces one process on the first device.

Standard torchrun launch:

```bash
torchrun --standalone --nproc-per-node=4 main.py \
  --config configs/mjepa_1d_base.yaml
```

`main.py` honors `RANK`, `WORLD_SIZE`, and `LOCAL_RANK`. CUDA training uses
NCCL; CPU integration tests use Gloo. `main_distributed.py` remains available
for Submitit/SLURM launches.

Checkpoints contain unwrapped encoder and predictor weights, the EMA target,
optimizer, optional AMP scaler, all schedules, completed iteration, next data
position, mask state, and per-rank Python/NumPy/PyTorch RNG states. Writes are
atomic. Every `checkpoint_interval` iterations, and at the final iteration,
training writes `<write_tag>-latest.pth.tar` and `<write_tag>-it<N>.pth.tar`.
Training metrics are written only to TensorBoard, using the global iteration
as the horizontal axis. CSV and JSONL training logs are no longer written.

Exact resume requires the same world size, dataset, batch size, seed, precision,
mask settings, and optimization configuration. It continues at the next batch,
including partway through a data pass, without loading already consumed samples.
Masks are generated in the training process so worker prefetching cannot advance
saved mask state. The current dataset loader is deterministic; worker count may
change on resume. Legacy epoch-training checkpoints cannot resume under this new
schedule; their compatible RoPE model weights remain readable by model-loading
utilities. To resume an iteration-based checkpoint:

```yaml
meta:
  load_checkpoint: true
  read_checkpoint: null  # null selects <write_tag>-latest.pth.tar
```

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
accumulated since the previous log event. Resume purges TensorBoard events
after the loaded checkpoint iteration. Checkpoints and the saved run configuration
remain available for reproducibility and resume.

## Raw-motion experiments

Experiments train directly on normalized motion frames. No pretrained motion
checkpoint or frozen motion-feature cache is used. Shared Transformer layers
and architecture sizes can still be used to initialize models from scratch.

- `experiment.classification`: 100STYLE single-label and BABEL multi-label
  classification with CNN, Transformer, or raw temporal-mean linear baselines.
- `experiment.segmentation`: supervised BABEL frame classification with
  a Transformer trained from scratch.
- `experiment.tmr`: raw-motion/text contrastive training. Frozen LLM2Vec
  sentence features feed a trainable RoPE-based JEPA text encoder; the JEPA motion
  encoder also trains from scratch. Both use the selected size preset.

```bash
python -m experiment.classification --model all
python -m experiment.classification --model linear
python -m experiment.segmentation --model-size tiny --describe
python -m experiment.segmentation --model-size base
python -m experiment.tmr.prepare_cache --device cuda:0
python -m experiment.tmr --model-size base --device cuda:0
```

Classification defaults to `dataset/100style-processed-nframes60-150`.
See the [classification](experiment/classification/README.md),
[segmentation](experiment/segmentation/README.md), and
[TMR](experiment/tmr/README.md) instructions for data and training options.

Frozen-feature probing, online downstream evaluation during JEPA training,
JEPA-initialized LoRA, and JEPA-conditioned prediction/generation tasks have
been removed. The main `train.py` continues to train JEPA itself and saves
ordinary training checkpoints and loss logs.

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
as `train/loss` in TensorBoard.
