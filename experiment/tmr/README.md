# Text-motion alignment

This experiment compares normalized raw MotionJEPA motion frames with frozen JEPA
tokens. Both branches learn independent text and motion alignment Transformers
with symmetric multi-positive InfoNCE. It implements the contrastive part of TMR;
there is no motion reconstruction decoder or VAE loss.

## Environment

Run preparation, training, evaluation, and tests in the `motion-jepa` conda environment:

```bash
conda activate motion-jepa
python -m pip install -r experiment/tmr/requirements.txt
```

The requirements add Transformers 5.1, TensorBoard, and a compatible Hugging Face
Hub version below 2. Existing PyTorch/CUDA packages remain in place.

## Model

Choose the frozen text backbone with `--text-model` in preparation and training:
`google/t5gemma-2b-2b-ul2` (the unchanged default, 2304-dimensional tokens), or
`distilbert/distilbert-base-uncased` (768-dimensional tokens). Local directories
of either model type are supported. The loader detects the architecture from its
config, freezes all parameters, and retains valid final hidden-state tokens;
it does not load a language-model or classification head. Head input widths are
inferred from the prepared features.

The default frozen text backbone is the first-generation, text-only pretrained
[`google/t5gemma-2b-2b-ul2`](https://huggingface.co/google/t5gemma-2b-2b-ul2).
`T5GemmaEncoderModel` loads with `is_encoder_decoder=False`; no decoder is
constructed. Standard Hugging Face loading can still download all weight shards
of the published encoder-decoder checkpoint. Accept the Gemma model license and
authenticate with `hf auth login`, or provide an already downloaded local model
directory using `--text-model`.

Preparation honors `HF_HOME`/`HF_HUB_CACHE` inherited from the activated
environment; the scripts do not override them. On this server, activating
`motion-jepa` sets `HF_HOME=/data/seokhyeon/hf-cache/`. Activate the environment
before each command so downloads and authentication use that existing cache.

The complete valid encoder token sequence is retained. Each trainable readout
projects its input to 256 dimensions, prepends its own learnable CLS token, adds
sinusoidal positions, and applies six pre-norm Transformer blocks by default (four
heads, 1024-wide FFN, dropout 0.1). `--text-depth` and `--motion-depth` independently
set the number of blocks in the trainable alignment heads. Each defaults to
`--depth` (6); an individual option overrides the shared value for that head only.
These options do not change the frozen text or JEPA backbones. By default,
each head's final LayerNorm CLS output is L2-normalized; CLS is added after the
frozen text encoder, not to its tokenizer input. `--text-pooling mean` and
`--motion-pooling mean` instead average valid tokens after the head's final
LayerNorm, then L2-normalize. A zero-depth head is supported only with mean
pooling: it projects the frozen tokens, applies LayerNorm, and pools without adding
positional embeddings or a Transformer block. The two pooling choices are saved
in checkpoints and checked on resume; historical checkpoints default to CLS.

Raw motion uses `[150,366]` frame features. JEPA uses the checkpoint's temporal
tokens, e.g. `[50,384]` for the current 3-frame patch encoder. For 2D checkpoints,
the spatial axis is averaged and time is retained. `target_encoder` is the default
checkpoint key; `--checkpoint-key encoder` is also supported. Valid-frame masks
are converted using the checkpoint token layout, excluding incomplete temporal
patches. An input with zero valid JEPA tokens is rejected.

JEPA tokens are standardized per channel using the mean and population standard
deviation of **all valid tokens in the captioned training split**. Longer clips
contribute more tokens; padding and excluded clips contribute none. Validation
and test use exactly these training statistics. Preparation computes moments in
FP64 in bounded chunks and saves FP32 `jepa/stats/mean.npy`, `std.npy`, and
`complete.json` under the cache root. Normalization is applied in FP32 when loading
tokens, using `(x - mean) / max(std, 1e-6)` before batch padding; the cached BF16
features retain the original encoder outputs. This is separate from `--stats-path`,
which supplies the raw-input statistics used before the JEPA encoder.

## BONES captions and caches

Preparation joins the existing processed BONES index to temporal annotation
filenames. Event boundaries use `floor(seconds * fps)` and half-open intervals.
Every event with at least one overlapping frame contributes a separate caption
candidate, ordered by start/end time and original annotation order. Whitespace is
normalized and identical descriptions are deduplicated across the clip, so repeats
do not increase their sampling weight. Descriptions are never concatenated.
Motion windows and their source-level splits stay unchanged. No-overlap clips are
excluded, without falling back to whole-motion captions.

For the current dataset this retains 191,037 train, 6,605 validation, and 13,067
test clips. Boundary event fragments are included, so a candidate can describe a
partially observed action. Candidates come from overlapping temporal events;
they may describe successive actions, rather than equivalent paraphrases.

Text features are extracted once per distinct candidate, using at most 256 tokens.
Each training access samples one candidate uniformly and loads only its token
sequence. PyTorch/worker seeds make this sampling reproducible across an
epoch-boundary resume. Validation and test have no caption sampling.
Preparation records truncations. Features use lazy BF16-bit `uint16` NumPy memory
maps: packed valid text and JEPA tokens plus sequence offsets/lengths. Training
reads only the requested rows and dynamically pads
text batches. It does not instantiate either frozen backbone.

Metadata records dataset/annotation/catalog hashes, text model/tokenizer revision,
token limit, JEPA checkpoint/key/layout, and normalization statistics. Incomplete
or stale caches fail explicitly; rebuild with `--recompute-features`. Keep the
same `--max-samples-per-split` for all preparation runs sharing a cache root.
Format 2 stores individual candidates. Legacy format-1 caches contain joined-text
features and cannot train with candidate sampling: re-run preparation for both
branches with `--recompute-features`, or use a new cache root. Old checkpoints and
their retrieval scores use the previous caption policy and should remain separate.

## Prepare and compare

Run from the repository root with `motion-jepa` activated. Use the same paired
cache, normalization statistics, seed, and head/training settings for both branches:

```bash
TMR_JEPA_CHECKPOINT=output/mot_patch_base_1d-p3-bs.512-ep.300-nframes150/motion-jepa-patch-1d-p3-latest.pth.tar
TMR_STATS=dataset/bones-seed-processed-nframes150/stats
TMR_CACHE=dataset/tmr-cache

python -m experiment.tmr.prepare_cache \
  --input-source raw --stats-path "$TMR_STATS" --cache-root "$TMR_CACHE"

python -m experiment.tmr.prepare_cache \
  --input-source jepa --jepa-checkpoint "$TMR_JEPA_CHECKPOINT" \
  --stats-path "$TMR_STATS" --cache-root "$TMR_CACHE"

python -m experiment.tmr \
  --input-source raw --stats-path "$TMR_STATS" --cache-root "$TMR_CACHE" \
  --output-root output/tmr/raw/seed-42

python -m experiment.tmr \
  --input-source jepa --jepa-checkpoint "$TMR_JEPA_CHECKPOINT" \
  --stats-path "$TMR_STATS" --cache-root "$TMR_CACHE" \
  --output-root output/tmr/jepa/seed-42
```

The default dataset is `dataset/bones-seed-processed-nframes150`, and annotations
are `dataset/bones-seed/metadata/seed_metadata_v002_temporal_labels.jsonl`.
`--dataset-root`, `--annotations-path`, `--text-model`, `--text-revision`, and
`--max-text-length` must agree between preparation and training. For JEPA,
normalization defaults to the checkpoint's pretraining statistics; for raw it
defaults to the processed dataset's statistics. Pass the same `--stats-path`
explicitly for a controlled comparison.

For a small smoke run, prepare **both** branches into a separate cache root with
`--max-samples-per-split 8`, then train with `--epochs 2 --warmup-epochs 0
--batch-size 4 --num-workers 0`. A smoke cache is identified in its manifest and
cannot silently be mistaken for the full dataset. It still needs text model
access, unless a local model is supplied.

## DistilBERT comparison

Use a separate TMR feature-cache root for each text model. This feature cache is
distinct from the shared Hugging Face download cache. A cache prepared for
T5Gemma cannot train a DistilBERT run, or vice versa; mismatched signatures and
resume configurations are rejected.

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate motion-jepa
TMR_TEXT_MODEL=distilbert/distilbert-base-uncased
TMR_CACHE=output/tmr-cache-distilbert

python -m experiment.tmr.prepare_cache \
  --input-source raw --text-model "$TMR_TEXT_MODEL" \
  --stats-path "$TMR_STATS" --cache-root "$TMR_CACHE"

python -m experiment.tmr.prepare_cache \
  --input-source jepa --text-model "$TMR_TEXT_MODEL" \
  --jepa-checkpoint "$TMR_JEPA_CHECKPOINT" \
  --stats-path "$TMR_STATS" --cache-root "$TMR_CACHE"

python -m experiment.tmr \
  --input-source raw --text-model "$TMR_TEXT_MODEL" \
  --stats-path "$TMR_STATS" --cache-root "$TMR_CACHE" \
  --output-root output/tmr/distilbert-raw/seed-42

python -m experiment.tmr \
  --input-source jepa --text-model "$TMR_TEXT_MODEL" \
  --jepa-checkpoint "$TMR_JEPA_CHECKPOINT" \
  --stats-path "$TMR_STATS" --cache-root "$TMR_CACHE" \
  --output-root output/tmr/distilbert-jepa/seed-42
```

Use the same model ID/revision and token limit for both preparation and training.
DistilBERT supports at most 512 positions; the default `--max-text-length 256`
is valid. Evaluation restores the selected model and feature cache from the
checkpoint automatically. To compare with T5Gemma, use its model ID with a
different feature-cache and output directory. All existing head-depth and
training-fraction options work with either backbone.

## Visualize cached motion-caption pairs

The viewer reads an existing `raw/paired-index.json` or `jepa/paired-index.json`
and its `prepared.json`. It displays the individual candidate captions next to
the original motion, with the source interval and clip duration. Selecting another
motion updates both together. Legacy caches still display their joined caption,
marked as a combined caption in the motion information. Playback, frame stepping,
speed, skeleton, mesh, and foot-contact controls are shared with the root viewer.

```bash
python -m experiment.tmr.visualize_dataset \
  --cache-root "$TMR_CACHE" --input-source raw --split train --port 6006
```

Use `--input-source jepa` to browse the JEPA cache's pairs; these also render the
associated original raw motion because JEPA tokens cannot be decoded to joints.
The viewer does not prepare data, extract text/JEPA features, load backbones, or
require Hugging Face authentication. It needs the original motion files referenced
by the cache. Use `--dataset-root` if that dataset has moved, `--mesh` for the
skinned body, `--limit 0` for all pairs, and `--sample-id` or `--sample-index` for
the initial selection. `python experiment/tmr/visualize_dataset.py` is also supported.

## Training and retrieval

Defaults: AdamW, LR `1e-4`, final LR `1e-6`, weight decay `0.01`, 100 epochs,
five-epoch warmup and cosine decay, batch 256, gradient clipping 1, temperature
0.1, seed 42, and CUDA BF16 autocast. Each motion is paired with one randomly
selected candidate. A sampled caption is positive for every motion that lists it
as a candidate, even if that motion sampled a different caption. Different-caption
clips from the same source with overlapping time intervals are excluded from
negatives. Semantically similar but
different captions remain negatives; no unvalidated text-similarity threshold is
used. A final singleton training batch is skipped because it has no negatives.

To compare motion-head capacity, keep the text head at six blocks and use a
one-block motion head for both inputs. Reuse the existing prepared cache and
compare these new runs with the default six-block runs:

```bash
python -m experiment.tmr \
  --input-source raw --stats-path "$TMR_STATS" --cache-root "$TMR_CACHE" \
  --text-depth 6 --motion-depth 1 --seed 42 \
  --output-root output/tmr/raw-motion-depth1/seed-42

python -m experiment.tmr \
  --input-source jepa --jepa-checkpoint "$TMR_JEPA_CHECKPOINT" \
  --stats-path "$TMR_STATS" --cache-root "$TMR_CACHE" \
  --text-depth 6 --motion-depth 1 --seed 42 \
  --output-root output/tmr/jepa-motion-depth1/seed-42
```

Use `--motion-depth 2` and new output directories for the two-block comparison.
Both commands retain the existing defaults of 100 epochs, batch 256, LR `1e-4`,
and full training data. Add `--train-fraction 0.1` to both for the 10% comparison,
using separate output directories for that fraction. No backbone feature
extraction is needed when changing head depth.

`config.json` and checkpoints record both resolved head depths. Standalone
evaluation restores these automatically. Historical checkpoints with only
`depth` still load and resume with that depth for both heads. Resume checks the
effective head depths; changing the shared fallback while explicitly retaining
both original head depths is allowed. Changing either head's depth requires a
new training run and output directory.

Use `--train-fraction` to compare training sizes while reusing the prepared cache:

```bash
python -m experiment.tmr \
  --input-source raw --stats-path "$TMR_STATS" --cache-root "$TMR_CACHE" \
  --train-fraction 0.1 --seed 42 --output-root output/tmr/raw/fraction-0.1/seed-42

python -m experiment.tmr \
  --input-source jepa --jepa-checkpoint "$TMR_JEPA_CHECKPOINT" \
  --stats-path "$TMR_STATS" --cache-root "$TMR_CACHE" \
  --train-fraction 0.1 --seed 42 --output-root output/tmr/jepa/fraction-0.1/seed-42
```

The default fraction is `1.0`; accepted values are `0 < fraction <= 1`.
The selected count is `floor(fraction * N)`, where N is the number of captioned
train clips in the prepared cache. Selecting fewer than two clips raises an error.
Only training is reduced: validation/test motions and caption galleries remain
complete for that cache. Selection samples motion clips, not source sequences or
individual caption candidates. An already limited smoke cache uses its own train
count as N and retains its existing validation/test limits.

`--seed` determines a fixed random permutation. Fractions take prefixes of that
same permutation, so smaller subsets are contained in larger ones. Raw and JEPA
use the same clip IDs when the paired cache, fraction, and seed match. Caption
sampling continues independently on each training access. The chosen IDs are
saved in `<output-root>/train-subset.json`; checkpoints fingerprint the selection,
and resume requires the same fraction and seed. Standalone evaluation restores
the selection from the checkpoint without extra arguments.

For JEPA, mean/std use only the selected training clips' valid tokens and are
applied to train, validation, and test. Subset statistics are cached separately
under `<cache-root>/jepa/stats-subsets/<signature>/`; full-cache statistics and
backbone features remain available for other training sizes. For a subset, run
`prepare_stats --train-fraction 0.1 --seed 42` to compute statistics separately,
adding `--recompute` to repair them. Use a different output directory for each
fraction/seed experiment. Raw input normalization continues to use `--stats-path`.

Validation and test use complete galleries, never minibatch retrieval. Text queries
and candidates are all unique candidate captions; motion queries and candidates
are all retained clips. Text-to-motion accepts any clip listing the query caption;
motion-to-text accepts any of the clip's candidates. Reported recalls
are fractions, ranks start at one, and score ties use ascending candidate index.
Source-overlap exclusions apply only during training. Similarity computation tiles
both axes to bound memory. Best-model selection uses the average of text-to-motion
and motion-to-text validation R@1; test is evaluated with that selected model.

The output directory contains `config.json`, `provenance.json`, `runtime.json`,
`metrics.csv`, `best.pth.tar`, `latest.pth.tar`, and `summary.json`. Checkpoints
contain alignment heads, optimizer/scheduler, and RNG/loader states, without
backbone weights. Resume an interrupted run by repeating its original command
with `--resume`; model, optimization (including planned epochs), and feature
provenance must match. A new experiment needs a new output directory.

TensorBoard is enabled by default and writes to `<output-root>/tensorboard`.
It records each optimizer step's loss and learning rate, epoch loss/sample count/
duration, all validation retrieval metrics, the best validation score/epoch, and
the selected checkpoint's test retrieval metrics. Scalars use optimizer steps
as their horizontal axis. `--resume` keeps committed events and purges events
after the saved step. Writers flush each epoch and close on completion or errors.
Use `--tensorboard-dir PATH` for another directory or `--no-tensorboard` to disable.

```bash
tensorboard --logdir output/tmr --port 6007
```

JEPA preparation automatically computes the training feature statistics. Existing
format-2 JEPA caches also gain them on the first training/evaluation load without
extracting any backbone features again. To compute them separately on an existing
cache, use:

```bash
python -m experiment.tmr.prepare_stats --cache-root "$TMR_CACHE"
```

The statistics manifest fingerprints training tokens, offsets, and mean/std;
checkpoints retain this provenance. Changed statistics cannot silently resume or
evaluate an existing run. Use `prepare_stats --recompute` to repair statistics;
if the underlying features changed, start a new training run. JEPA checkpoints
trained before this normalization was added require a new run. Raw checkpoints
remain compatible.

```bash
python -m experiment.tmr.evaluate \
  --checkpoint output/tmr/jepa/seed-42/best.pth.tar --split test \
  --output output/tmr/jepa/seed-42/test-retrieval.json

python -m unittest discover -s tests -p 'test_tmr*.py'
```

Tests use a small local text encoder/tokenizer and synthetic motion/checkpoints,
without Hugging Face downloads. A real pretrained 2B inference check additionally
requires access to the gated model. Cache preparation reports an actionable error
if that access is unavailable.


## TMR v2: JEPA-sized raw-motion training

Select v2 with `--model-version v2 --model-size tiny`. The motion encoder is the
actual frame-token `MotionTransformer1D`, randomly initialized and fully trained
with the existing TMR contrastive objective. Its RoPE uses frame times in seconds,
theta 100, and time scale 1. There are no context/target masks, predictor, or EMA
teacher; padding masks still exclude invalid frames.

| Size | Width | Blocks | Heads |
|---|---:|---:|---:|
| tiny | 192 | 6 | 3 |
| small | 256 | 8 | 4 |
| base (v2 default) | 384 | 12 | 6 |
| large | 512 | 16 | 8 |
| huge | 768 | 24 | 12 |
| giant | 1024 | 32 | 16 |

Motion encoding is raw normalized frames → JEPA input projection and RoPE blocks
→ terminal LayerNorm → valid-frame mean → L2 normalization. There is no extra
motion projection or CLS token. Frame count and FPS come from validated dataset
metadata; batches are zero-padded to that frame count.

The frozen text backbone remains selectable between T5Gemma and DistilBERT.
Text encoding is frozen tokens → trainable linear projection to the selected
JEPA width → trainable per-token LayerNorm → masked mean → L2 normalization.
There are no additional text transformer blocks. Both modalities therefore have
the selected size's embedding width.

After preparing a raw-motion cache with the chosen text backbone:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate motion-jepa
python -m experiment.tmr \
  --model-version v2 --model-size tiny \
  --input-source raw \
  --text-model google/t5gemma-2b-2b-ul2 \
  --cache-root dataset/tmr-cache \
  --output-root output/tmr-v2/tiny/seed-42
```

Use the existing DistilBERT cache and its matching `--text-model` to change text
backbones. All Hugging Face paths continue to follow the activated environment's
cache settings. Existing `--train-fraction`, optimization, logging, resume, and
evaluation options apply. Defaults remain 100 epochs, batch size 256, AdamW LR
1e-4, weight decay 0.01, five warmup epochs, cosine decay to 1e-6, temperature
0.1, gradient clipping 1, BF16, and seed 42. Validation mean bidirectional R@1
selects the checkpoint before test-gallery evaluation.

V1 remains the default and old checkpoints remain readable. V2 accepts raw
motion only, rejects JEPA checkpoints/features and explicitly supplied legacy
architecture flags (`--depth`, `--text-depth`, `--motion-depth`, `--embed-dim`,
`--num-heads`, `--ff-dim`, `--dropout`, and pooling flags). Its architecture
is determined by the named size. Use `--resume` with the same v2 configuration;
evaluation reconstructs the model from the checkpoint automatically.

Without `--output-root`, v2 writes to `output/tmr-v2/<size>/seed-<seed>`, or
`output/tmr-v2/<size>/frac<fraction>/seed-<seed>` for subsets. Specify distinct
output roots when comparing text backbones or optimization settings.
`config.json` stores the version, resolved size, frame count, FPS, and RoPE
settings. `parameters.json` and `summary.json` report trainable motion, text,
and total parameter counts; frozen text-backbone parameters are excluded.


## RoPE JEPA + LoRA for TMR

Use v2 with `--motion-adaptation lora --jepa-checkpoint <final-EMA-checkpoint>`.
The encoder is initialized from the chosen `--checkpoint-key` (default
`target_encoder`). Raw frames are encoded online so adapters receive gradients;
frozen JEPA token caches are not used. No additional motion transformer is added.

Only attention QKV/output LoRA adapters and the text projection/LayerNorm train.
Original JEPA parameters and the text backbone stay frozen. Both readouts remain
valid-token mean pooling and L2 normalization at the JEPA embedding width.
LoRA defaults are rank 8, alpha 16 and dropout 0, applied in every block.
The existing TMR optimizer LR and schedule apply to both adapters and text readout.

The model size is inferred from the checkpoint; an explicit `--model-size` must
match. Only non-patchified 1D RoPE checkpoints with theta 100/time scale 1 are
supported. Frame count, FPS, input width and the original raw normalization
statistics must match the prepared dataset. Both T5Gemma and DistilBERT caches
are supported through their existing text-model/cache options.

Checkpoint provenance includes the source SHA256, selected encoder key,
pretraining epoch and complete source config. Resume rejects changed initialization
or recipes. Training checkpoints contain the frozen encoder plus adaptation and
optimizer states; standalone evaluation reconstructs the LoRA architecture and
verifies the original source identity. Parameter reports distinguish trainable
motion adapters/text readout from frozen motion parameters.

Example (with the matching raw-motion/T5 cache prepared):

```bash
python -m experiment.tmr --model-version v2 --motion-adaptation lora --model-size base \
  --input-source raw --jepa-checkpoint <RoPE-JEPA-base-final-epoch300.pth.tar> \
  --checkpoint-key target_encoder --cache-root output/tmr-cache-rope-raw \
  --dataset-root dataset/bones-seed-processed-nframes150 \
  --stats-path dataset/bones-seed-processed-v2-nframes150-min60/stats \
  --output-root output/tmr-jepa-lora/base/seed-42
```

Use an explicit output root when comparing pretrained checkpoints. Default LoRA
outputs use `output/tmr-jepa-lora/<size>/seed-<seed>` (or `checkpoint-size`
when size is inferred), with fraction subdirectories for partial training data.

Prepare a raw cache manifest with the checkpoint's original normalization before training.
The shared text bank may be reused through a symlink when its signed text provenance
matches. The launched base experiment uses output/tmr-cache-rope-raw for this purpose.
