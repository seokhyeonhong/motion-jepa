# Motion shortcut generation

Run from the repository root in the motion-jepa conda environment. This experiment
uses MotionTransformer1D components with generation-specific AdaLN-Zero blocks;
it does not load JEPA weights into the generator.

## Training

Choose experiment/repa/configs/local.yaml (365D) or global.yaml (366D).
Both use prepared Bones-SEED train/validation splits, 150-frame capacity and 30 FPS.
Teacher paths are deliberately required when alignment is enabled.

    # Flow-only baseline
    python -m experiment.repa train --config experiment/repa/configs/local.yaml --output output/repa/local-flow --alignment-weight 0 --shortcut-fraction 0

    # Shortcut baseline without JEPA
    python -m experiment.repa train --config experiment/repa/configs/local.yaml --output output/repa/local-shortcut --alignment-weight 0

    # Shortcut plus raw MSE alignment (default)
    python -m experiment.repa train --config experiment/repa/configs/local.yaml --output output/repa/local-mse --teacher-checkpoint /path/to/checkpoint.pth.tar --teacher-config /path/to/pretraining.yaml

    # Cosine comparison: identical setup, independent output directory
    python -m experiment.repa train --config experiment/repa/configs/local.yaml --output output/repa/local-cosine --teacher-checkpoint /path/to/checkpoint.pth.tar --teacher-config /path/to/pretraining.yaml --alignment-loss cosine

Use torchrun --standalone --nproc_per_node=2 -m experiment.repa train ...
for DDP. Batch size is per rank. CPU/Gloo is supported with --device cpu.
A selected teacher must have the same representation and frame capacity. Its input
statistics can differ: training inputs are denormalized and renormalized
for the teacher. Targets come from the final block before final LayerNorm by
default. Three-frame teachers use pooled groups of student frame features and
discard incomplete/padded groups.

Raw MSE preserves teacher feature magnitude. Cosine is an explicit ablation.
Alignment is computed only on flow samples. The teacher is frozen;
projector and student train.

The shortcut base resolution is 128. Flow samples use step size 1/128; shortcut
samples use powers of two from 1 through 1/64 and two EMA half-step predictions.
This finite-step convention follows https://github.com/kvfrans/shortcut-models .
The motion path uses exact Gaussian-to-data interpolation and no image-specific
clipping. EMA decay is 0.999. No text/class conditioning or classifier-free guidance.

## Resume and validation

Repeat the original training command and overrides, adding
--resume output/repa/local-mse/latest.pt. Resume validates config, world
size, statistics and teacher identity and restores optimizer/scheduler, per-rank
random state and sampler position. Use --max-steps 2 for a bounded additional
two-step check without changing the learning-rate schedule.

    python -m experiment.repa evaluate --checkpoint output/repa/local-mse/latest.pt --output output/repa/local-mse/validation.json

Validation uses a fixed subset and fixed random draws. Logged losses, feature
norms, cosine similarity and consistency by step size diagnose optimization;
they are not generation-quality metrics. TensorBoard logs are in each run's
tensorboard directory. Milestone checkpoints and latest.pt contain the resolved
configuration, statistics and teacher provenance.

## Sampling

    python -m experiment.repa sample --checkpoint output/repa/local-mse/latest.pt --output output/repa/local-mse/samples-4 --steps 4 --count 16 --length 150 --fps 30 --seed 0

Supported step counts: 1, 2, 4, 8, 16, 32, 64, 128 (up to configured base resolution).
Sampling loads only the generator EMA, not the JEPA teacher or alignment projector.
Outputs are motions.npy (denormalized [B,T,C], padded rows zero) and
metadata.json (valid lengths, FPS, representation, seed, checkpoint, evaluations,
and sampling latency). Length/FPS specify structural layout, not semantic conditioning.
Latency includes integration calls and excludes checkpoint loading and export.

## Tests

    python -m pytest experiment/repa/test_generation.py model/test_patchify_1d.py
