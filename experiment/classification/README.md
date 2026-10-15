# Raw-motion classification

Train CNN, Transformer, or temporal-mean linear classifiers from scratch.
100STYLE uses cross entropy; BABEL uses multi-label BCE. Every model reads
normalized motion frames and masks padding. Statistics come from the input
dataset's training split.

```bash
python -m experiment.classification --model all --model-size base
python -m experiment.classification --model linear
python -m experiment.classification --model transformer --dataset-root dataset/babel-120-processed-nframes30-150
```

The default dataset is `dataset/100style-processed-nframes60-150`.
Use `--dataset-root` for a processed dataset, `--output-root` for checkpoints,
and `--findings-root` for summaries. `--model all` runs CNN and Transformer.
Use `--help` for optimization settings and resume options.

Selection uses validation accuracy or validation mAP when validation exists.
Without validation, the final epoch is selected; test scores never select
a checkpoint. No pretrained encoder, feature extraction, or probing is used.

## Transformer sizes

The classifier uses the same `MotionTransformer1D` encoder and `MODEL_SPECS`
as JEPA, supervised segmentation, and TMR V2, with a trainable classification
head over the mean of valid frame tokens. RoPE uses each sample's FPS,
theta 100, and time scale 1. Initialization, QKV biases, MLP ratio 4,
LayerNorm epsilon 1e-6, and zero dropout/drop-path match the shared encoder.
There is no CLS token or absolute position embedding.

| Size | Width | Blocks | Heads |
| --- | ---: | ---: | ---: |
| tiny | 192 | 6 | 3 |
| small | 256 | 8 | 4 |
| base (default) | 384 | 12 | 6 |
| large | 512 | 16 | 8 |
| huge | 768 | 24 | 12 |
| giant | 1024 | 32 | 16 |

```bash
python -m experiment.classification --model transformer --model-size small \
  --output-root output/100style-classifiers-small
```

Use a separate output root for each size. Resolved architecture details are
saved in checkpoints and run signatures; the previous fixed-size classifier's
checkpoints cannot be resumed as this model. `--model-size` applies only to
the Transformer, including when running `--model all`.
