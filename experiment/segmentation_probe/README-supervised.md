# JEPA-sized supervised raw-motion segmentation

Train the actual frame-token JEPA encoder from scratch with a per-frame linear BABEL120 classifier. Select tiny/small/base/large/huge/giant directly from model.specs.MODEL_SPECS. The encoder preserves JEPA initialization, MLP ratio 4, QKV biases, LayerNorm epsilon, zero dropout/drop-path, and time-based RoPE (theta 100, scale 1). No predictor, EMA branch, masking, pooling, or encoder projection is added. All encoder and classifier parameters train.

Example (activate conda motion-jepa first):

    python -m experiment.segmentation_probe.supervised --model-size base --output-root output/segmentation-supervised-jepa/base/seed-42 --describe

Remove --describe to run six LR/weight-decay tuning trials followed by selected refit. Existing findings/022 data caches, source-disjoint split, normalization, masked BCE, and metrics are reused without changing the prior experiments. AdamW LRs 1e-4/3e-4/1e-3, WD .01/.1, 100 epochs per trial, warmup 5, evaluation every 5, batch 256, seed 42, BF16 train / FP32 evaluation, gradient clip 1. Checkpoints record resolved architecture and recipes; source or recipe changes reject resume. This runner retains the earlier empirical class-prior head bias.

Encoder parameter counts match JEPA for identical input dimensions. Total adds (width+1)*120 classifier parameters. Counts exclude JEPA's predictor and duplicate EMA encoder.

Full training is a separate submission: inspect GPU and pod memory limits and available resources first, especially for large/huge/giant. Existing helpers keep full normalized raw-frame caches on the device.
