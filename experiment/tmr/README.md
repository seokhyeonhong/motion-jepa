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
There is one supported architecture: LLM2Vec plus RoPE-based JEPA encoders.
Architecture is selected only by `--model-size`; checkpoints store the resolved
size settings and dataset timing without a model version. Runs are saved under
`output/tmr/<size>/seed-<seed>` (with a fraction subdirectory for train subsets).

The text backbone follows `/source/seokhyeon/kimodo/kimodo/kimodo/model/llm2vec/llm2vec_wrapper.py`.
The trainable encoders use this repository's shared JEPA implementation.
