# 100STYLE Classification and Linear-Probe Handoff

This document consolidates the 100STYLE preprocessing, Motion-JEPA linear probing, supervised baselines, and JEPA-token classifier support implemented in this repository. Existing metrics are historical artifacts; no experiment was rerun to produce this document.

## Data and preprocessing

- Source BVHs: `dataset/100STYLE_soma77/bvh/*_soma77.bvh`.
- 100STYLE is already on the standard SOMA skeleton. Its preprocessing path therefore does not call `from_standard_tpose`, which is intended for converting non-standard input skeletons.
- The approximately 60 FPS input is sampled at a fixed step to 30 FPS and converted to the repository's 366-dimensional `motion_jepa_366_v1` representation on the 30-joint SOMA skeleton.
- Each source motion is divided into 90-frame windows with stride 90. Windows never overlap, and an incomplete final window is discarded. Canonicalization is performed independently for each complete window.
- All window descriptors are shuffled globally with split seed 42 and divided 80%/10%/10%. The resulting counts are 20,916 train, 2,615 validation, and 2,614 test windows (26,145 total) from 810 source motions.
- A sample ID has the form `<source-name>_<window-index>`. `index.json` records `style`, `motion_code`, `source_id`, `start_frame`, and `end_frame`.
- `StyleLabelIndex` reads styles from `index.json`, validates records and duplicate IDs, sorts all discovered style names, and fixes the class-to-index mapping for every split. The label is the filename-derived style name with the final motion code removed.
- Raw-motion classifiers use the 100STYLE training-split mean and standard deviation. JEPA-based experiments instead use the BONES-SEED pretraining statistics referenced by the JEPA checkpoint; they do not normalize with 100STYLE statistics.

The split is intentionally window-level. Disjoint windows from one recording may occur in multiple splits. Results therefore measure style classification on unseen windows from recordings whose recording-specific characteristics may already occur in training. They must not be interpreted as generalization to entirely unseen recordings.

The preprocessing entry point is `dataset/preprocess_100style.py`. For example:

```bash
python dataset/preprocess_100style.py \
  --limit 10 \
  --workers 1 \
  --output dataset/100style-processed-preview
```

`--limit` limits source BVH files, not generated windows. The full processed dataset used by these experiments is `dataset/100style-soma77-processed`.

## Package layout

The classification implementation lives under `experiment/linear_probe/`:

- `dataset.py`: style indexing, motion datasets, normalization, and cached token datasets.
- `probe.py`: checkpoint/statistics loading, frozen encoder feature extraction and caching, and linear-head training.
- `lr_sweep.py`: resumable all-checkpoint learning-rate sweep and report generation.
- `cnn.py`: temporal ResNet classifier.
- `transformer.py`: frame Transformer classifier with a learnable CLS token.
- `supervised.py`: shared raw-motion and JEPA-token classifier training CLI.

## Motion-JEPA linear probe

The probe reconstructs the checkpoint's 1D Motion-JEPA model and strictly loads the EMA `target_encoder` by default. The encoder remains in evaluation mode with gradients disabled. Motions are normalized using the checkpoint's pretraining mean and standard deviation. The encoder output `[B,T,D]` is reduced by the mean over valid frame tokens, without L2 normalization or an extra LayerNorm. Pooled float32 features, labels, and sample IDs are cached separately for train, validation, and test.

The only trainable component is a biased `nn.Linear(feature_dim, num_classes)`. It uses ordinary cross entropy, SGD with momentum 0.9, zero weight decay, cosine LR decay, batch size 256, and 50 epochs. The best validation overall top-1 epoch is restored before one test evaluation.

The completed sweep covers the latest checkpoint in each of 15 direct child run directories under `output/`, seven learning rates (`0.001`, `0.003`, `0.01`, `0.03`, `0.1`, `0.3`, `1.0`), and seeds 0, 1, and 2. All learning rates are reported; the sweep itself does not declare an optimum. Feature caches are stored in `output/<run>/linear-probe/features/`, and per-combination artifacts are under `findings/000-linear-probe-lr-sweep-all-latest/runs/`.

Single-probe example:

```bash
python -m experiment.linear_probe \
  --checkpoint output/<run>/motion-jepa-1d-latest.pth.tar \
  --dataset-root dataset/100style-soma77-processed
```

Sweep entry point:

```bash
python -m experiment.linear_probe.lr_sweep --device cuda:0
```

Each cache records hashes for the checkpoint, dataset index, and statistics, together with the checkpoint key, model name, and pooling definition. A stale cache is rejected unless feature recomputation is explicitly requested.

## Raw-motion supervised baselines

Both raw baselines consume `[B,90,366]` and are trained end to end with ordinary cross entropy. The common recipe is AdamW, LR `3e-4`, weight decay `0.05`, batch size 256, five epochs of linear warmup followed by cosine decay over 100 epochs, gradient clipping at 1.0, seed 42, and CUDA BF16 autocast with float32 parameters and optimizer state. There is no augmentation, class weighting, or balanced sampler. The best validation overall top-1 checkpoint is restored for a single test evaluation.

### Temporal ResNet CNN

`MotionCNNClassifier` uses a Conv1d stem from 366 to 256 channels with GroupNorm and GELU. Its three stages have widths 256, 384, and 512, with two residual blocks per stage and stride-2 temporal downsampling at stage transitions. It applies masked global mean pooling, dropout 0.1, and a linear class head. The recorded model has 6,371,172 parameters.

### Transformer

`MotionTransformerClassifier` projects each frame from 366 to dimension 256 and uses eight pre-norm Transformer blocks, eight attention heads, MLP ratio 4, dropout 0.1, and stochastic depth increasing to 0.1.

Two architecture generations have results:

- The historical run in `findings/001-supervised-100style-classifiers/` used masked mean pooling over valid frame tokens. It has 6,461,284 parameters.
- The current implementation prepends a learnable CLS token, uses a learnable position embedding for CLS plus up to 90 frames, and classifies the final CLS representation. Its completed raw-motion run is under `output/100style-classifiers-cls/transformer/seed-42/` and has 6,461,796 parameters.

Raw training example:

```bash
python -m experiment.linear_probe.supervised \
  --input-source raw \
  --model all \
  --dataset-root dataset/100style-soma77-processed \
  --seed 42 \
  --device cuda:0
```

## CNN and Transformer with JEPA token input

The same classifiers now accept a generic `input_dim`, while `motion_dim` remains a compatibility alias. In JEPA mode, a frozen 1D Motion-JEPA encoder produces frame tokens `[B,90,D]`. Split caches store BF16 tokens, valid lengths, labels, sample IDs, and cache metadata. No additional feature normalization is applied after encoding.

The encoder is removed from GPU memory after extraction; only the CNN or Transformer is optimized. The CNN retains temporal residual processing and masked mean pooling. The Transformer applies its own input projection, positional embedding, and CLS-token pooling to the JEPA tokens.

Default paths are:

- Token cache: `<JEPA checkpoint directory>/linear-probe/token-features/{train,val,test}.pt`
- Classifiers: `<JEPA checkpoint directory>/linear-probe/classifiers/{cnn,transformer}/seed-42/`
- Findings: `<JEPA checkpoint directory>/linear-probe/classifiers/findings/`

Classifier best/latest checkpoints, `summary.json`, and `model-config.json` record the resolved JEPA checkpoint path and SHA256, checkpoint key, JEPA model name, feature dimension, frame count, pretraining statistics paths and hashes, token-cache dtype, and cache metadata version. JEPA weights are not duplicated inside classifier checkpoints. Feature extraction uses the fixed `--feature-batch-size`; adaptive OOM retry logic was deliberately removed to keep the implementation simple.

Example:

```bash
python -m experiment.linear_probe.supervised \
  --input-source jepa \
  --jepa-checkpoint output/<run>/motion-jepa-1d-latest.pth.tar \
  --checkpoint-key target_encoder \
  --model all \
  --dataset-root dataset/100style-soma77-processed \
  --seed 42 \
  --device cuda:0
```

This JEPA-token classifier mode is implemented and tested synthetically, but no full real-data CNN or Transformer result has been recorded yet.

## Metrics

- Overall top-1: the fraction of all windows whose highest-scoring class is correct. Styles with more windows contribute more.
- Macro accuracy: compute accuracy independently for each style, then average styles equally. This exposes performance on less frequent styles.
- Top-5: the fraction of windows whose correct style appears among the five highest-scoring classes.

## Consolidated results

| Experiment | Selection / pooling | Test top-1 |
|---|---|---:|
| Motion-JEPA linear probe, `mot_giant_1d` | LR 0.3, best validation-selected LR; valid-token mean | 90.61 +/- 0.19% |
| Original `mot_base_1d` linear probe | LR 0.3, best validation-selected LR; valid-token mean | 75.38 +/- 0.31% |
| Raw temporal ResNet CNN | Best validation epoch; masked mean | 94.30% |
| Raw Transformer, historical | Best validation epoch; masked mean | 95.14% |
| Raw Transformer, current | Best validation epoch; CLS token | 93.11% |

The `mot_giant_1d` LR 0.3 aggregate has validation top-1 `90.73 +/- 0.06%` and test top-1 `90.61 +/- 0.19%`. LR 1.0 has a slightly higher test mean (`90.72%`) but a lower validation mean, so it is not used for the validation-selected comparison. The supervised numbers are single seed-42 runs and should not be read as mean-plus-standard-deviation estimates.

On this window split, the raw supervised CNN and both raw Transformers outperform the strongest frozen mean-pooled JEPA probe. This does not by itself show that the representation is poor: a single linear layer after global temporal averaging cannot recover information discarded by pooling, while the supervised models learn temporal aggregation end to end. The CLS-token change also did not improve this run; the current CLS Transformer scored 2.03 percentage points below the historical mean-pooled Transformer on test top-1.

Full tables and plots are available in the [linear-probe sweep report](../findings/000-linear-probe-lr-sweep-all-latest/README.md) and the [historical supervised report](../findings/001-supervised-100style-classifiers/README.md).

## I-JEPA context and next experiments

I-JEPA itself does not use a CLS token during pretraining. Its official representation evaluation averages patch tokens, and its supplementary evaluation also considers concatenating averaged outputs from the last four encoder layers. Published I-JEPA results compare linear probing and full fine-tuning, but do not provide a directly matched raw-input supervised classifier baseline for this motion setting. For example, I-JEPA ViT-H/16 at 448 resolution reports 81.1% linear evaluation and 87.1% full fine-tuning on ImageNet-1K, a 6.0-point gap. See the [I-JEPA paper](https://arxiv.org/abs/2301.08243) for the image-domain context.

Recommended follow-ups are:

1. Probe the concatenated mean-pooled outputs of the last four JEPA layers, matching the richer I-JEPA evaluation protocol.
2. Train the implemented CNN and CLS Transformer on frozen JEPA frame tokens and compare them with the pooled linear head. This tests whether temporal information is present but hidden by early mean pooling.
3. Fine-tune the pretrained encoder, first with a small encoder LR and then end to end, while keeping validation-based model selection.
4. Add a source-recording-level split to measure generalization to genuinely unseen recordings and compare it with the current window-level result.
5. Repeat supervised baselines over multiple seeds before treating small differences between CNN and Transformer architectures as stable.

