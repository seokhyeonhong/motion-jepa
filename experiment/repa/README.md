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

    python -m pytest experiment/repa/test_generation.py experiment/repa/test_quality.py

## Periodic generation-quality evaluation

The configs enable Kimodo TMR-FID and foot-skating evaluation every 10,000 training
iterations, independently of the loss-validation interval. Configure:

    generation_evaluation:
      enabled: true
      interval: 10000
      samples: 1024
      batch_size: 32
      sampling_steps: [128]
      seed: 42
      checkpoint_dir: null
      on_completion: false

Set sampling_steps to [1, 2, 4, 8, 32, 128] to compare shortcut sampling budgets;
each adds a complete generation pass. Keep [128] for the flow-only baseline.
Use enabled: false to disable quality evaluation. on_completion optionally adds
evaluation at the configured final training iteration; bounded --max-steps checks
do not implicitly trigger it.

Install evaluation dependencies in the motion-jepa environment:

    python -m pip install -r experiment/repa/requirements-evaluation.txt

The evaluator automatically downloads only the motion encoder/config/statistics
from nvidia/TMR-SOMA-RP-v1, pinned to revision
e427752ae3446dedba49e928c93ddc9f0e413401. No text encoder or JEPA teacher is needed.
For offline use, checkpoint_dir may point to an existing snapshot containing
config.yaml, last_weights/motion_encoder.pt, and stats/motion. These assets are
already cached on space-01. Weights retain their separate upstream model license.

FID compares generated EMA samples with a fixed validation subset using Kimodo's
deterministic, unit-normalized TMR motion embeddings. It is not image Inception
FID or text-motion FID. Both real and generated motions are denormalized and
decoded through their 365D/366D representation into SOMA30 FK positions. TMR
preprocessing canonicalizes heading and XZ origin, computes its 186 features,
selects the checkpoint's global-root/body statistics, and normalizes with
sqrt(std**2 + 1e-5). The released evaluator requires 30 FPS.

Reported metrics:
- fid: generated-vs-reference TMR Frechet distance.
- foot_skate_from_height: mean toe speed in m/s when toe height is below 0.05 m.
- foot_skate_from_pred_contacts: mean foot speed in m/s on predicted contacts.
- foot_skate_max_vel: per-motion maximum contacting foot speed, averaged over motions.
- foot_skate_ratio: fraction of contacting toe transitions exceeding 0.2 m/s;
  both endpoints must be below 0.05 m.

Foot metrics use actual sequence lengths and Kimodo's per-motion aggregation.
They also report real-data reference scores. A motion with no qualifying contacts
receives zero for contact-conditioned metrics, matching upstream behavior.
Compare runs using the same sample count, subset seed, encoder and sampling budget;
small-sample FID is only a diagnostic and is not an official Kimodo benchmark score.

Results go to <run-output>/quality/step-XXXXXXXX.json and TensorBoard:
quality/steps_128/fid, quality/steps_128/foot_skate_ratio, etc.
JSON records the subset indices, evaluator weight/statistics hashes and upstream
commit. Real features are cached in memory for subsequent evaluations.
Evaluation runs in FP32 with TF32 disabled, restores RNG/model state, and uses
fixed noise independent of training. In DDP, rank zero evaluates the full sample
distribution while other ranks wait; errors are propagated to all ranks.

You may change only generation_evaluation settings when resuming an existing
checkpoint; the remaining training configuration must still match. A running
Python process does not reload edited YAML files.

Standalone evaluation (also works for older checkpoints without evaluation config):

    python -m experiment.repa quality --checkpoint /path/to/latest.pt --config experiment/repa/configs/local.yaml --output output/repa/quality-check --device cuda

The --config override supplies only generation_evaluation settings. Architecture,
representation and normalization always come from the checkpoint.
The existing evaluate command continues to report training-objective losses.

Upstream code, license, modifications and provenance are documented in
kimodo_eval/NOTICE.md. Tests:

    python -m pytest experiment/repa/test_generation.py experiment/repa/test_quality.py
