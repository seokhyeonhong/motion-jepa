# Raw-motion segmentation

Train a frame-token Transformer from scratch on BABEL's 120-class frame labels.
The model uses the shared size presets and RoPE layers. It does not load a
pretrained encoder or import scripts from previous findings.

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
the same dataset, architecture, and recipe. Findings are saved under
`findings/segmentation/<model-size>/seed-<seed>` by default.
This is a standalone run; the old findings-specific hyperparameter sweep and
extra feature standardization are no longer part of this task.
