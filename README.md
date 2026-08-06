# Motion-JEPA

Motion-JEPA learns motion representations by predicting target embeddings from
masked context embeddings. The repository contains a self-contained BONES-SEED
preprocessing pipeline, the `motion_jepa_366_v1` SOMA30 representation, two
Motion-JEPA transformer variants, distributed pretraining, and visualization.

The code does not require the Ardy or Kimodo repositories at runtime.

## Layout

```text
dataset/          preprocessing, fixed-clip dataset, and loader
model/            frame-token and skeletal-temporal transformers
mask/             deterministic structured mask collators
motion_rep/       366-D representation and geometry
skeleton/         BVH parsing, SOMA skeletons, kinematics, and assets
visualization/    viser dataset viewer, skeleton renderer, and skinning
utils/            distributed, logging, scheduler, and tensor helpers
configs/          `_1d` and `_2d` pretraining configurations
train.py          shared JEPA training loop
main.py           local-device and torchrun entry point
```

Public imports are intentionally top-level:

```python
from dataset import MotionDataset, make_motion_dataset
from mask import MaskCollator1D, MaskCollator2D
from model import MotionTransformer1D, MotionTransformer2D
from motion_rep import MotionJEPAMotionRep
from skeleton import SOMASkeleton30, SOMASkeleton77, parse_bvh_motion
from visualization import MotionJEPADatasetViewer, SOMASkin
```

## Installation

Python 3.10 or newer and PyTorch 2.1 or newer are recommended.

```bash
pip install -r requirements-motion.txt
```

## Dataset preprocessing

The default preprocessing command reads BONES-SEED SOMA Uniform BVH files and
writes independently canonicalized clips as individual NumPy arrays:

```bash
python dataset/preprocess_dataset.py --workers 64
```

The default output is `dataset/bones-seed-processed`:

```text
bones-seed-processed/
├── motions/{train,val,test}/*.npy
├── motions/{train,val,test}.json
├── train.txt
├── val.txt
├── test.txt
├── index.json
├── meta.json
├── errors.jsonl
└── stats/{mean,std}.npy
```

Each motion file stores one raw `float32[length,366]` array. Split rows contain
`sample_id,relative_npy_path,fps,actual_length`. Higher source rates are reduced only by exact
fixed-step indexing. Lower or non-divisible rates are reported in
`errors.jsonl`. Complete windows use 50% overlap by
default. After the last complete window, one uncovered tail is retained when
it has at least `--min_frames` frames (90 by default).

The final output directory is created immediately, and each sequence NPY is
saved there as soon as its source motion is converted. Split manifests,
statistics, and metadata are finalized after conversion. Complete output is
reused unless `--overwrite` is supplied; interrupted partial output requires
`--overwrite` to restart.

The loader validates `meta.json`, manifests, every split row, value size, FPS, feature
dimension, finite values, and normalization statistics. Runtime resampling is
not supported. Variable-length tails are normalized over their real frames,
then end-padded with literal zeros; padding is excluded from masks, attention,
targets, and loss.

Workers convert and save source motions directly, avoiding transfer of large
feature arrays back to the parent process. Training uses global per-epoch
randomization through `DistributedSampler`; each loader worker lazily opens
only the NPY files selected for its current batch. NPY is the supported
processed dataset format for training and visualization.

Visualize processed clips with:

```bash
python visualize_dataset.py dataset/bones-seed-processed --split train --mesh
```

## Model variants

`_1d` uses one token per 366-D frame and temporal transformer attention. Its
context and target masks are contiguous temporal blocks.

`_2d` routes the representation into a `[frames,30 joints]` grid. Each joint
receives its own position, rotation, and velocity fields; root position and
heading go to the root token, while the four contact values go to their
corresponding foot/toe tokens. Encoder blocks apply temporal attention per
joint followed by spatial attention per frame. The predictor retains separate
context and target-query streams, including when their coordinates overlap.

Named factories are available for `tiny`, `small`, `base`, `large`, `huge`,
and `giant`, for example `mot_base_1d` and `mot_base_2d`. There are no
unsuffixed compatibility aliases.

## Training

Batch size is per rank. Learning rates are not automatically scaled.

Single GPU:

```bash
python main.py --fname configs/mjepa_1d.yaml --devices cuda:0
```

Local multi-GPU:

```bash
python main.py \
  --fname configs/mjepa_2d.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3
```

`main.py` spawns one process per configured device and propagates child
failures. `--debug` forces one process on the first device.

Standard torchrun launch:

```bash
torchrun --standalone --nproc-per-node=4 main.py \
  --fname configs/mjepa_1d.yaml
```

`main.py` honors `RANK`, `WORLD_SIZE`, and `LOCAL_RANK`. CUDA training uses
NCCL; CPU integration tests use Gloo. `main_distributed.py` remains available
for Submitit/SLURM launches.

Checkpoints contain unwrapped encoder and predictor weights, the EMA target,
optimizer, optional AMP scaler, all schedules, epoch/global step, mask state,
and per-rank Python/NumPy/PyTorch RNG states. Writes are atomic. Exact resume
requires the same world size and continues from the latest completed epoch:

```yaml
meta:
  load_checkpoint: true
  read_checkpoint: null  # null selects <write_tag>-latest.pth.tar
```

BF16 autocast is controlled by `meta.use_bfloat16`. BF16 does not use a
gradient scaler; optional FP16 training uses `meta.use_float16` and restores
its scaler state.

TensorBoard logging is enabled in the standard configs. Install TensorBoard
and launch it against the training output:

```bash
pip install tensorboard
tensorboard --logdir output
```

Events are written under `<logging.folder>/tensorboard` at `logging.log_freq`.
Loss, iteration time, learning rate, and weight decay are interval averages
accumulated since the previous log event. The CSV continues to store raw values
for every optimization step.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite covers preprocessing, representation parity, data validation,
semantic 2D routing, both encoder/predictor variants, masking and overlap,
checkpoint restoration, launch dispatch, single-process resume, and a real
two-process Gloo DDP run. This checkout exposes one GPU, so NCCL multi-GPU
execution cannot be exercised locally.

## Attribution

Motion-JEPA includes modified Apache-2.0 portions and assets derived from NVIDIA
Ardy and Kimodo. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).

The JEPA training design is based on Meta's I-JEPA implementation and paper:

```bibtex
@article{assran2023self,
  title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author={Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2301.08243},
  year={2023}
}
```
