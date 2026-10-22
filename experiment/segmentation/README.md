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
