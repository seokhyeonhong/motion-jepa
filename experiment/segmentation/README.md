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
Validation BABEL-120 frame mAP selects the best epoch; test is evaluated after
selection. BABEL-60 subset scores use the same supervised frame population.

Outputs include latest/best checkpoints, configuration, epoch metrics, and a
summary. `--resume` restores optimizer, scheduler, and RNG state and requires
the same dataset, architecture, and recipe. Findings are written only when
`--findings-root` is explicitly supplied; use a sequential numbered directory
when requesting a findings report (for example, `findings/001-segmentation-base`).

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
