# 100STYLE Classification and Linear-Probe Handoff

This document describes the current 100STYLE linear-probe and raw-motion classifier workflows. The unified report at `findings/000-100style-classification/` is generated only from the current seed-42 artifacts.

## Data protocol

- Dataset: `dataset/100style-soma77-processed`.
- Input representation: 90 frames by 366 motion features.
- Split: 20,916 train, 2,615 validation, and 2,614 test windows.
- Split unit: non-overlapping windows shuffled with seed 42. Windows from one source recording may occur in multiple splits.
- Labels: 100 style names resolved through `StyleLabelIndex` and recorded in `class-index.json`.
- Raw-motion classifiers use the 100STYLE training statistics. JEPA experiments use the pretraining statistics referenced by each checkpoint.

The reported numbers measure classification on the existing window-level split. They are single-seed results and do not support variance or unseen-recording claims.

## Package layout

The implementation lives under `experiment/linear_probe/`:

- `dataset.py`: style indexing and raw/cached-feature datasets.
- `features.py`: frozen encoder reconstruction, extraction, and cache validation.
- `train_probe.py`: one linear-probe run.
- `lr_sweep.py`: all-checkpoint learning-rate sweep and component report.
- `cnn.py`: temporal ResNet classifier.
- `transformer.py`: Transformer classifier with a learnable CLS token.
- `train_classifier.py`: raw-motion and JEPA-token classifier training.
- `report.py`: unified findings generator.

## Motion-JEPA linear probes

The probe strictly loads the pretrained EMA target encoder by default and keeps it frozen in evaluation mode. Valid output tokens are globally mean-pooled and cached as float32 features. The only trainable layer is a biased `nn.Linear(feature_dim, 100)`.

The current sweep evaluates the latest checkpoint from each of 15 direct children of `output/`, learning rates `0.1`, `0.3`, `0.5`, `0.7`, and `1.0`, and seed 42. Every head is trained for 50 epochs with SGD, momentum 0.9, zero weight decay, cosine decay, and batch size 256. The training epoch is selected by validation top-1. The unified comparison then selects each checkpoint's LR by validation top-1, breaking exact ties in favor of the lower LR.

Run one probe:

```bash
python -m experiment.linear_probe \
  --checkpoint output/<run>/motion-jepa-1d-latest.pth.tar \
  --dataset-root dataset/100style-soma77-processed
```

Run the complete sweep:

```bash
python -m experiment.linear_probe.lr_sweep --device cuda:0
```

The default sweep findings directory is `findings/000-100style-classification/linear-probe`. Feature caches remain under each JEPA run's `linear-probe/features/` directory.

## Raw-motion classifiers

Both classifiers consume `[B,90,366]` windows and are trained end to end with ordinary cross entropy. The shared recipe is AdamW, LR `3e-4`, weight decay `0.05`, batch size 256, five warmup epochs, cosine decay over 100 epochs, gradient clipping at 1.0, BF16 autocast, and seed 42. There is no augmentation, class weighting, or balanced sampler.

`MotionCNNClassifier` is a temporal ResNet with widths 256, 384, and 512, two residual blocks per stage, masked mean pooling, dropout 0.1, and 6,371,172 parameters.

`MotionTransformerClassifier` projects frames to dimension 256 and uses eight pre-norm blocks, eight attention heads, MLP ratio 4, dropout 0.1, and stochastic depth up to 0.1. It always prepends a learnable CLS token and classifies the final CLS representation. There is no pooling option. The model has 6,461,796 parameters.

Run both raw-motion classifiers:

```bash
python -m experiment.linear_probe.train_classifier \
  --input-source raw \
  --model all \
  --dataset-root dataset/100style-soma77-processed \
  --seed 42 \
  --device cuda:0
```

Outputs default to `output/100style-classifier-raw`, and component findings default to `findings/000-100style-classification/classifiers`.

The validation-best checkpoint is restored and evaluated again on validation and test. `classifier-best.pth.tar` contains only its format version, model state, and reconstruction architecture. An incomplete run's `classifier-latest.pth.tar` additionally contains optimizer, scheduler, RNG, progress, best-score, and signature state; it is removed when training completes. Final metrics and provenance are stored once in `summary.json`, with no `model-config.json`.

## JEPA-token classifiers

The same CNN and CLS Transformer can train on frozen target-encoder frame tokens `[B,90,D]`. Tokens are cached as BF16 without extra feature normalization, after which the encoder is released and only the classifier is optimized.

```bash
python -m experiment.linear_probe.train_classifier \
  --input-source jepa \
  --jepa-checkpoint output/<run>/motion-jepa-1d-latest.pth.tar \
  --checkpoint-key target_encoder \
  --model all \
  --dataset-root dataset/100style-soma77-processed \
  --seed 42 \
  --device cuda:0
```

Token caches and classifier outputs default below the JEPA checkpoint directory. This mode is implemented and covered by synthetic tests; the unified report contains only the completed raw-motion classifiers and pooled linear probes.

## Current validation-selected results

| Method | Selection | Validation top-1 | Test top-1 |
|---|---|---:|---:|
| CLS Transformer | epoch 71 | 94.84% | 94.84% |
| CNN | epoch 76 | 94.26% | 94.45% |
| `mot_giant_1d-bs.512-ep.300` probe | LR 0.3, epoch 48 | 90.52% | 90.74% |

The giant probe is the strongest probe by validation top-1. On test top-1, the CNN exceeds it by 3.71 percentage points and the CLS Transformer by 4.09 points. These comparisons do not isolate encoder quality from the probe's global mean-pooling bottleneck.

Regenerate and inspect the complete report with:

```bash
python -m experiment.linear_probe.report
```

The [unified findings report](../findings/000-100style-classification/README.md) contains all 15 validation-selected probes, both classifiers, all five learning rates per probe, plots, and machine-readable results.
