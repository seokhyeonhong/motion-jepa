# Text-motion retrieval from raw motion

Both trainable modalities use the repository's RoPE-based MotionJEPA encoder:

- Text: frozen LLM2Vec/Llama-3-8B-Instruct with MNTP and supervised adapters
  → one 4096D sentence feature → trainable JEPA text encoder → 256D retrieval embedding.
- Motion: normalized raw frames → trainable JEPA motion encoder → 256D retrieval embedding.

Both branches reuse `MotionTransformer1D` from `model/motion_transformer_1d.py`,
including its linear input projection, pre-norm Transformer blocks, GELU,
4x feed-forward width, LayerNorm epsilon 1e-6, initialization, and zero dropout
defaults. RoPE uses theta=100 and time_scale=1. Motion positions are frame
times in seconds using dataset FPS; text positions are token indices.
Valid encoder outputs are mean-pooled, projected to 256D (identity for hidden
width 256), and L2-normalized. Each branch trains from scratch.

The symmetric multi-positive contrastive objective is unchanged.

## Size presets

`--model-size` controls **both** trainable encoders using `model/specs.py`.
Depth, hidden width and head count follow the JEPA size preset, while
retrieval embeddings stay 256D.

| Size | Hidden width | Layers per modality | Heads |
| --- | ---: | ---: | ---: |
| tiny | 192 | 6 | 3 |
| small | 256 | 8 | 4 |
| base (default) | 384 | 12 | 6 |
| large | 512 | 16 | 8 |
| huge | 768 | 24 | 12 |
| giant | 1024 | 32 | 16 |

## Prepare and train

Run in the `motion-jepa` conda environment:

```bash
python -m pip install -r experiment/tmr/requirements.txt
python -m experiment.tmr.prepare_cache --device cuda:0
python -m experiment.tmr --model-size base --device cuda:0
```

The default text sources match Kimodo:

- MNTP: `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp`
- Supervised adapter: `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised`

The loader merges MNTP into its Llama-3 base, then loads the supervised adapter.
Hugging Face access to `meta-llama/Meta-Llama-3-8B-Instruct` is required.
It honors the activated environment's Hugging Face cache. Override sources
with `--text-model` and `--text-adapter`; pin them with `--text-revision`
and `--text-adapter-revision`. Use the same options for cache preparation and training.

Sentence extraction uses bidirectional attention, the Llama-3 user template,
and mean pooling that excludes the instruction prefix. As in Kimodo, each
caption is encoded separately for BF16 reproducibility; `--text-batch-size`
controls cache write chunks. An explicit noncausal attention mask supports
the pinned Transformers 5.1 API. LLM2Vec produces one feature token per caption;
the trainable JEPA text encoder processes this token at position zero.

The default input is `dataset/bones-seed-processed-nframes150-min60`.
Caption candidates are selected by temporal overlap. Training samples captions;
evaluation uses a fixed caption and a complete retrieval gallery.
Motion normalization uses dataset training statistics.

The default cache is `output/tmr-cache-llm2vec`. Training reads completed features and raw motion
files without loading the 8B backbone. Model, adapter, revisions, pooling policy,
tokenizer and extraction settings are recorded in cache metadata.
Use `--dataset-root`, `--annotations-path`, and `--cache-root` consistently.

## Evaluate and resume

```bash
python -m experiment.tmr.evaluate --checkpoint output/tmr/base/seed-42/best.pth.tar
python -m experiment.tmr --model-size base --device cuda:0 --resume
python -m experiment.tmr.visualize_dataset --cache-root output/tmr-cache-llm2vec
python -m unittest experiment.tmr.test_architecture -v
```

Training saves configuration, provenance, metrics, and latest/best checkpoints.
Resume requires matching settings and data provenance.
Supervised runs use LLM2Vec plus RoPE-based JEPA encoders.
Their architecture is selected by `--model-size`; checkpoints store the resolved
size settings and dataset timing without a model version. Runs are saved under
`output/tmr/<size>/seed-<seed>` (with a fraction subdirectory for train subsets).

The text backbone follows `/source/seokhyeon/kimodo/kimodo/kimodo/model/llm2vec/llm2vec_wrapper.py`.
The trainable encoders use this repository's shared JEPA implementation.


## Attentive probing

Use a frozen JEPA checkpoint to extract one frame feature per selected block:

```bash
python -m experiment.tmr --mode probe --model-size base \
  --feature-source jepa --feature-layer 3 --encoder-key target_encoder \
  --checkpoint output/mot_base_1d-bs.512-it.200000-nframes60-150_allctx/motion-jepa-1d-it200000.pth.tar \
  --output-root output/tmr/probe-layer3/seed-42

python -m experiment.tmr --mode probe --model-size base --feature-source raw \
  --output-root output/tmr/probe-raw/seed-42
```

`--feature-layer` is a one-based transformer block index. Features come from
that block's output before the encoder's final LayerNorm. JEPA extraction uses
its checkpoint's pretraining input normalization. The raw baseline uses the
same attention architecture directly on raw motion frames.

The entire trainable motion transformer is replaced by one multihead attention
operation with a learned query and one linear projection to the 256D retrieval
space. The existing text encoder architecture remains trainable and unchanged;
both outputs are L2-normalized and use the existing multi-positive contrastive
loss. `--probe-heads` defaults to 6 and must divide the input feature width.

Per-channel population mean and standard deviation are fitted only on valid
frames of the actual captioned training clips, including any selected training
fraction. Validation/test features use those same fixed statistics. These small
normalization vectors are buffers in the probe checkpoint.

Motion features are precomputed into anonymous CPU float32 tensors, one layer
per process. No motion feature file or memory-mapped cache is written. All
splits remain in RAM during training, and the frozen JEPA network is released
from the GPU before the trainable model is constructed. Base features for the
full prepared corpus require about 72 GiB of RAM. `--precompute-batch-size`
controls extraction batches; probe training and evaluation use zero loader
workers to avoid replicating the bank. Existing LLM2Vec sentence caches are
read without loading the frozen text backbone or creating a new cache.

Resume and standalone evaluation rebuild the transient motion features, verify
the checkpoint/data provenance, and restore the saved normalization and probe
weights. Neither frozen JEPA weights nor frame feature banks are included in
probe checkpoints. Supervised mode remains the default.


## Iteration-based training

Training follows the main JEPA trainer's global-update convention. Defaults are
`--iterations 150000`, `--warmup-iterations 7500`, `--batch-size 256`,
`--eval-interval 1500`, `--checkpoint-interval 1500`, and `--log-freq 120`.
Warmup covers 5% of the default budget; validation and periodic checkpointing
each run every 1%. Logging intervals scale with the original iteration budgets.
These values never scale with `--train-fraction` or dataset length. One iteration
is one optimizer update. The shared `utils.schedulers.WarmupCosineSchedule`
advances before each update, with `--start-lr 0`, peak `--lr 1e-4`, and
`--final-lr 1e-6`. Weight decay and the remaining task settings stay fixed.

```bash
python -m experiment.tmr --mode probe --feature-source raw \
  --train-fraction 0.01 --seed 42 --iterations 150000 \
  --warmup-iterations 7500 --eval-interval 1500 \
  --checkpoint-interval 1500 --batch-size 256 \
  --output-root output/tmr/probe-raw-iterations/frac0.01/seed-42
```

The trainer runs a fixed number of iterations using a continuous stream of
shuffled batches. Workers persist when num_workers is positive, and prefetching
continues across dataset passes. Incomplete tails are dropped to keep every
optimizer batch the same size; the subset is reshuffled on the next pass.
RAM feature probing retains its zero-worker loader. A subset smaller
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

Training captions use the existing random sampling. Resume does not reproduce
the interrupted data order exactly. Legacy TMR checkpoints remain supported for
standalone evaluation.


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

TMR cache preparation also accepts `--representation`. Use a separate
`--cache-root` for the local dataset and prepare it before training, for example:

```sh
python -m experiment.tmr.prepare_cache --dataset-root /path/to/local-babel --cache-root /path/to/local-text-cache --representation LocalMotionJEPARep
python -m experiment.tmr --dataset-root /path/to/local-babel --cache-root /path/to/local-text-cache --representation LocalMotionJEPARep --mode supervised --output-root /path/to/new-tmr-run
```


The prepared local datasets on this workspace can be selected directly with
`--representation LocalMotionJEPARep`; standard dataset paths (and TMR's standard
cache path) switch to their local counterparts. Explicit custom paths are
preserved and validated. Local runs use a separate `local-motion-jepa` default
output subdirectory. Omit the option or use `MotionJEPAMotionRep` for the original
global defaults.

### Raw-motion temporal patches

Supervised TMR accepts `--patchify` (default: `--no-patchify`). It uses the
JEPA Conv1d motion tokenizer with kernel/stride 3 and no padding: 150 raw frames
produce 50 tokens, with floor(valid_frames / 3) complete patches. Incomplete
trailing patches are excluded from attention and mean pooling. Text tokenization,
raw-frame normalization, data splits, and optimization settings are unchanged.
Use `--representation LocalMotionJEPARep` for local inputs. The default patched
output path includes `patchify/`; use distinct `--output-root` values for explicit
comparison paths. Checkpoints save the option, and evaluation restores it
automatically. Legacy checkpoints default to frame-wise tokenization.
This option applies only to supervised raw-motion TMR, not frozen JEPA probes.
