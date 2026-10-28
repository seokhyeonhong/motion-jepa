# Raw-motion segmentation

Train a frame-token Transformer from scratch on BABEL's 120-class frame labels.
The default supervised model uses the shared size presets and RoPE layers.
A separate attentive-probe mode evaluates frozen intermediate JEPA frame features
or raw motion with the same shallow frame classifier.

```bash
python -m experiment.segmentation --model-size tiny --describe
python -m experiment.segmentation --model-size base --device cuda:0
```

The default dataset is `dataset/babel-segmentation-120-processed-nframes30-150`.
Use `--dataset-root` to change it. Normalization uses that dataset's training
statistics. Dataset validation enforces source-disjoint splits.

Training uses masked BCE on annotated frames, AdamW, warmup and cosine decay.
Padding and unannotated frames do not contribute to loss or metrics.
Validation BABEL-120 frame mAP selects the best iteration; test is evaluated after
selection. BABEL-60 subset scores use the same supervised frame population.

Outputs include latest/best checkpoints, configuration, iteration metrics, and a
summary. `--resume` restores optimizer, scheduler, and RNG state and requires
the same dataset, architecture, and recipe. Findings are written only when
`--findings-root` is explicitly supplied; use a sequential numbered directory
when requesting a findings report (for example, `findings/segmentation/000-base`).

## Frozen attentive probing

```bash
python -m experiment.segmentation --mode probe --model-size base \
  --feature-source jepa --feature-layer 6 --checkpoint /path/to/jepa-checkpoint.pth.tar \
  --output-root output/segmentation/probe-layer-6 --device cuda:0
python -m experiment.segmentation --mode probe --feature-source raw \
  --output-root output/segmentation/probe-raw --device cuda:0
```

`--feature-layer` selects a one-based encoder block output. `--encoder-key`
defaults to the frozen EMA `target_encoder`; `encoder` selects the online encoder.
JEPA inputs use the checkpoint's pretraining normalization. Intermediate outputs
are standardized using per-channel population mean/std recomputed over all valid
frames in this task's training split, including unannotated context frames.
Validation/test data and padding never contribute to these statistics. Raw motion
uses the same training-frame feature standardization.

The frame probe is one self-attention layer with a residual connection, followed
by a single linear class projection. It preserves one output per frame; it has no
feed-forward transformer block or trainable pretrained encoder. `--probe-heads`
defaults to six, compatible with both 384D base features and 366D raw motion.

Preprocessing reads each sample once and keeps features, labels, and masks only in
CPU RAM. `--precompute-batch-size` defaults to 256. The frozen encoder is released
from the GPU before the probe is created. Training uses zero DataLoader workers
to avoid duplicating the RAM cache. Checkpoints store probe parameters, feature
normalization buffers and source provenance, never per-sample features. Resuming
recomputes the RAM cache and requires identical feature source, layer and recipe.


## Limited training data

Add `--train-fraction 0.1` or `--train-fraction 0.01` to supervised or probe mode.
The fraction is measured over the existing training windows, rounded down.
Validation remains the existing complete evaluation set used for checkpoint
selection and reporting; the empty test split and all source split files remain
unchanged. No new validation or test partition is created.

```bash
python -m experiment.segmentation --mode probe --feature-source raw \
  --train-fraction 0.01 --seed 42 --device cuda:0
python -m experiment.segmentation --mode probe --feature-source jepa \
  --checkpoint /path/to/jepa-checkpoint.pth.tar --feature-layer 6 \
  --train-fraction 0.1 --seed 42 --device cuda:0
```

Selection reads validated labels without reading motion arrays. It first visits
categories from rarest to most common, randomly selecting a window for each
uncovered category. All labels in that window count toward coverage. A seeded
permutation fills the remaining budget without replacement. The coverage core
and permutation are independent of the fraction, so 1% is contained in 10% for
the same seed. Raw probes, JEPA layers, and supervised models select identical
IDs for the same fraction/seed. Missing training categories or a budget smaller
than the selected coverage core raise an error; the budget is never enlarged
silently. Coverage requires at least one supervised positive frame per category,
not equal category frequencies.

Subset selection happens before motion feature extraction. Raw/JEPA probe feature
statistics use only valid frames of selected training clips. JEPA input statistics
stay fixed to pretraining. Supervised runs preserve their existing dataset-wide
training input statistics; these fractions restrict training examples, not access
to those precomputed statistics. At fraction 1, the existing full-data path is
retained without scanning labels for subset selection.

Each fractional run writes `train-subset.json` with IDs, seed, requested/actual
fraction, coverage-core size, and per-category positive window/frame counts.
The same metadata is part of checkpoint signatures and checked on resume.
Default output paths include `frac0.1` or `frac0.01`; an explicit `--output-root`
is used exactly as supplied and must be different for each experiment.

The existing `experiment.probe_suite` also accepts `--train-fraction`, forwards it
to all five runs with seed 42, and adds the fraction to output/report directory
names. Run it separately for each fraction with distinct sequential report
numbers. Reports include the training fraction and link the saved subset IDs;
raw-probe deltas require the same fraction and subset hash. Iteration budgets and LR schedules remain fixed across fractions. Historical
epoch runs remain readable but are not treated as matched iteration comparisons.


## Iteration-based training

Training follows the main JEPA trainer's global-update convention. Defaults are
`--iterations 30000`, `--warmup-iterations 1500`, `--batch-size 128`,
`--eval-interval 300`, `--checkpoint-interval 300`, and `--log-freq 300`.
Warmup covers 5% of the default budget; validation and periodic checkpointing
each run every 1%. Logging intervals scale with the original iteration budgets.
These values never scale with `--train-fraction` or dataset length. One iteration
is one optimizer update. The shared `utils.schedulers.WarmupCosineSchedule`
advances before each update, with `--start-lr 0`, peak `--lr 3e-4`, and
`--final-lr 3e-6`. Weight decay and the remaining task settings stay fixed.

```bash
python -m experiment.segmentation --mode probe --feature-source raw \
  --train-fraction 0.01 --seed 42 --iterations 30000 \
  --warmup-iterations 1500 --eval-interval 300 \
  --checkpoint-interval 300 --batch-size 128 \
  --output-root output/segmentation/probe-raw-iterations/frac0.01/seed-42
```

The trainer runs a fixed number of iterations and restarts its shuffled DataLoader
when exhausted. Training
uses `drop_last=True` to keep every optimizer batch the same size; incomplete
tails are dropped and the subset is reshuffled on the next pass. A subset smaller
than the requested batch size is rejected rather than silently shrinking batches.
Choose a common batch size that fits the smallest fraction. Validation/test loaders
retain all samples, including their final partial batches. `--eval-batch-size`
controls evaluation batching independently (default 256).

Validation always runs at the final iteration, even when the total is not a
multiple of its interval. Best checkpoints are selected only from validation.
Latest checkpoints are saved at the checkpoint interval, on a new validation
best, and at the final update. Metrics use `iteration`; summaries record
`global_step`, `best_step`, `samples_seen`, and `equivalent_epochs`. Equivalent
epochs mean sample exposures divided by selected training-set size; they do not
control the schedule.

Resume restores model/optimizer, scheduler, RNG, completed iteration count, and
unfinished training-metric accumulators, then starts a fresh shuffled data pass.
Checkpoints have no format-version field. Epoch CLI flags have been removed;
legacy epoch checkpoints have different training settings and cannot resume into
an iteration run. Historical metrics
remain readable by `experiment.probing_report` with an explicit epoch axis.

The suite launcher accepts `--iterations`, `--warmup-iterations`, `--eval-interval`,
`--checkpoint-interval`, `--log-freq`, and `--batch-size` overrides. Omitting them
uses the task defaults above; use identical values for every compared fraction.


### Motion representations

Both supervised training and frozen JEPA probing accept datasets preprocessed
with `MotionJEPAMotionRep` (366D) or `LocalMotionJEPARep` (365D).
The representation and input width are read from the dataset's `meta.json`.
Optionally pass `--representation LocalMotionJEPARep` to check that the selected
dataset uses the intended format. This option validates existing features; it
does not convert a global dataset into local coordinates.

Use a separate dataset directory, its own training normalization statistics,
and a new output directory for each representation. A frozen JEPA checkpoint
must have been pretrained with the same representation as the task dataset;
mismatches are rejected before feature extraction. Older checkpoints without
a representation field are interpreted as the original global representation.

Raw attentive probes require a head count dividing the feature width:
use `--probe-heads 5` for 365D local raw features. JEPA feature probes use the
encoder embedding width, so their usual six heads remain valid.

```sh
python -m experiment.segmentation --dataset-root /path/to/local-segmentation --representation LocalMotionJEPARep --mode supervised --output-root /path/to/new-segmentation-run
```


The prepared local datasets on this workspace can be selected directly with
`--representation LocalMotionJEPARep`; standard dataset paths (and TMR's standard
cache path) switch to their local counterparts. Explicit custom paths are
preserved and validated. Local runs use a separate `local-motion-jepa` default
output subdirectory. Omit the option or use `MotionJEPAMotionRep` for the original
global defaults.

### Motion depth override

For raw supervised segmentation, add `--model-size base --motion-depth 6`
to use six motion Transformer blocks with Base width 384 and six attention heads.
Any positive block count is accepted. Omitting the option preserves the model-size
preset. Explicit depths get separate default output directories (e.g.
`base-depth-6`), and the effective depth is saved in the checkpoint architecture.
This option does not apply to the attentive probe head. Existing strict resume
checks still require matching architecture, source hashes, and training settings.

### Raw motion patchification

Add `--mode supervised --model-size base --patchify` (optionally
`--motion-depth 6`). A stride-3/kernel-3 Conv1d embeds consecutive frames;
a linear head predicts three separate 120-class frame logits per token.
150 frames become 50 tokens and return 150 frame predictions. Labels,
supervision masks, loss and evaluation populations remain frame-level.

Invalid input frames are zeroed. A patch is active when any constituent frame
is valid, retaining supervision on incomplete final patches; fully padded
patches are masked. Non-multiple-of-three input lengths are padded internally
and cropped after decoding. This differs from retrieval's complete-patch-only
policy, intentionally preserving all segmentation frames. The policy is saved
in the architecture signature. Defaults remain frame-level, dropout 0.0 and
weight decay 0.01. Patch runs have a separate default output suffix `-patch`.

### Additional validation metrics

Macro-F1 averages per-class frame F1 over the full vocabulary (120 or 60),
with undefined F1 set to zero. Predictions use sigmoid >= 0.5.
Segment F1@10/25/50 uses temporal IoU thresholds 0.10/0.25/0.50,
one-to-one same-class matching, and micro aggregation across classes.
Each class has independent intervals, allowing overlapping BABEL labels.
Intervals are evaluated within each processed window; ignored frames break
intervals, and neither padding nor window boundaries join intervals.
These are window-level multilabel metrics, not stitched source-sequence scores.

Re-evaluate an existing raw supervised best checkpoint without changing history:

```sh
python -m experiment.segmentation.evaluate_checkpoint path/to/best.pth.tar
```

The evaluator verifies the checkpoint's dataset hashes and writes
`extended-validation-metrics.json` beside the checkpoint. Use
`--dataset-root` if the same dataset resides at a different mount path.

### Training data loading

Training uses a continuous stream of shuffled batches. Every dataset pass still
drops its incomplete final batch, but prefetching continues across passes.
Workers persist when num_workers is positive; probe mode remains single-process.
This changes the trainer code signature, so old checkpoints do not pass the
existing strict resume-signature check.


### Concatenated frozen patch JEPA features

Use the one-block Base-width segmentation decoder with an ordered layer list:

```bash
python -m experiment.segmentation.train \
  --mode probe --feature-source jepa --probe-architecture patch-transformer \
  --checkpoint output/mot_base_1d-bs.512-it.200000-nframes60-150_allctx_patch/motion-jepa-1d-it200000.pth.tar \
  --encoder-key target_encoder --feature-layers 3 6 9 12 \
  --representation MotionJEPAMotionRep \
  --dataset-root dataset/babel-segmentation-120-processed-nframes30-150 \
  --train-fraction 1.0 --seed 42 --iterations 30000 --warmup-iterations 1500 \
  --eval-interval 3000 --checkpoint-interval 3000 --batch-size 128 \
  --eval-batch-size 256 --precompute-batch-size 128 --num-workers 8 \
  --lr 0.0003 --final-lr 0.000003 --weight-decay 0.01 \
  --output-root output/segmentation/global-patch-jepa-concat-3-6-9-12/seed-42
```

Set the dataset and checkpoint paths for the current server. Pretraining input
statistics are resolved from the checkpoint configuration, as with existing JEPA
probes; that statistics directory must also be accessible.

- Layers are one-based and concatenated in the requested order. Use
  `--feature-layers 3 6` for two layers or `--feature-layers 3` for one.
  The legacy `--feature-layer 3` also works; the two flags are mutually exclusive.
  Empty, duplicate, nonpositive, and out-of-range layer selections are rejected.
- The frozen EMA encoder is evaluated once per batch, through the highest
  requested layer. Features are captured before the encoder's final LayerNorm.
  This decoder requires a three-frame patch checkpoint. All cached features
  remain in CPU RAM, and the frozen backbone is released before head training.
- Each layer/channel uses its own mean/std fitted only on valid training patch
  tokens. Concatenation changes channels, never temporal order: four Base layers
  give `[B,50,1536]`, projected to `[B,50,384]`.
- The trainable decoder has an input projection, exactly one Base-width
  MotionTransformer1D block (384 width, six heads, RoPE, MLP ratio 4), final
  LayerNorm, and a 384-to-360 frame classifier. There is no second patchification.
  Every token emits three ordered 120-label frame logits.
- Invalid raw frames are zeroed after input normalization. A patch is active
  when any of its frames is valid, preserving partial-patch tail supervision.
  Fully padded patches and invalid output frames remain masked. This matches
  the completed single-layer patch segmentation experiments.
- Dropout remains 0.0; weight decay defaults to 0.01. The original attentive
  frame-probe path remains the default. The raw supervised `--patchify` and
  `--motion-depth` flags do not apply to this decoder.
- The ordered layer list and fusion protocol are saved in checkpoint feature
  provenance and the summary. Resume requires the same list/order, source,
  dataset, and settings; it does not reinterpret old checkpoints.

For 120 labels and Base JEPA features, trainable parameter counts are:

| Selected layers | Concatenated width | Trainable parameters |
| --- | ---: | ---: |
| One | 384 | 2,061,672 |
| Two | 768 | 2,209,128 |
| Four | 1536 | 2,504,040 |

The frozen JEPA backbone is excluded. All widths retain the same 384-wide
Transformer block; only the input projection grows.


### LoRA fine-tuning of JEPA

Use `--mode lora --checkpoint PATH` for the entire JEPA encoder with rank-8
attention LoRA followed directly by a single linear segmentation head.
Original JEPA weights remain frozen; LoRA adapters and the linear head train.
No intermediate concatenation, MLP, extra Transformer or feature-statistics
calibration is used. `--mode probe` retains all existing frozen probing options.

```sh
python -m experiment.segmentation.train --mode lora --lora-rank 8 \
  --checkpoint output/mot_base_1d-bs.512-it.200000-nframes60-150_allctx_patch/motion-jepa-1d-it200000.pth.tar \
  --output-root output/segmentation/global-patch-lora-linear/seed-42
```

The final encoder output includes JEPA's original LayerNorm. A patch checkpoint
uses Linear(width, 3*120), reshaped to frame-level logits; a frame checkpoint uses
Linear(width, 120). Layout and depth are read from the checkpoint. Omit
feature-layer and probe-architecture arguments for LoRA. Intermediate layers or
concatenations are rejected. Global/local dataset representation must match JEPA.

`--lora-rank` defaults to 8; `--lora-alpha` defaults to rank (scale=1).
Adapters target fused attention QKV and output projections in every JEPA block.
B initializes to zero, preserving initial JEPA outputs. There is no adapter
dropout; frozen-backbone dropout stays disabled. Inputs retain checkpoint
pretraining normalization, and padded frames are masked.

Training is online, using persistent training workers when num-workers > 0.
Checkpoints include frozen weights, adapters, linear head, input statistics,
optimizer and RNG state. Strict resume checks provenance, rank, alpha and recipe.
Parameter reports separate trainable adapters/head from frozen weights.
For Base patch rank8: 221184 adapter + 138600 head = 359784 trainable parameters.


### Frozen patch-feature MLP probe

Use `--mode probe --probe-architecture mlp --model-size base --motion-depth 2`
with a patch JEPA checkpoint and `--feature-layer 3` or
`--feature-layers 3 6 9 12`. Hidden width follows tiny/small/base (192/256/384).
`motion-depth` counts Linear+GELU stages and defaults to 2. Each patch is
processed independently, then a linear head produces 3x120 frame logits;
there is no temporal pooling or extra attention. Partial patches retain valid
frame supervision, padded outputs are masked, and train-only feature statistics
are checkpointed. Existing BCE-with-logits training remains unchanged. This
option requires frozen patch JEPA features; LoRA remains a separate mode.
