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
