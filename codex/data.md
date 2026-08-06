# Dataset Preprocessing Summary

## Purpose

The preprocessing pipeline converts BONES-SEED SOMA Uniform BVH files into
the canonicalized 366-dimensional representation consumed by Motion-JEPA.
The implementation is self-contained under the top-level `motion_rep/`,
`skeleton/`, and `visualization/` packages and does not
require Ardy or Kimodo at runtime.

The entry point is:

```bash
python dataset/preprocess_dataset.py
```

## Default Paths

```text
BVH source:    dataset/bones-seed/soma_uniform/
Metadata:      dataset/bones-seed/metadata/seed_metadata_v004.csv
Split lists:   dataset/Kimodo-Motion-Gen-Benchmark/splits/
Output:        dataset/bones-seed-processed/
```

All paths can be overridden with command-line arguments.

## Processing Flow

For each split entry, the pipeline:

1. Parses local joint rotations, root translations, and frame time from BVH.
2. Converts root translations from centimeters to meters.
3. Rounds FPS using half-up rounding: `floor(source_fps + 0.5)`.
4. Requires the rounded source FPS to equal or be divisible by the configured
   FPS (60 by default). Lower and non-divisible sources are excluded.
5. Selects frames with `frames[::source_fps // configured_fps]`; no
   interpolation is performed.
6. Divides the result into complete configured windows, then retains the one
   uncovered remainder when it has at least `--min_frames` frames (default 90).
   A shorter qualifying source is saved whole.
7. Applies the SOMA77 standard T-pose transformation and selects the SOMA30
   joint subset.
8. Independently computes and canonicalizes the current Motion-JEPA
   representation for each window, including positions, rotations,
   velocities, and contacts.
9. Saves raw valid-frame arrays directly from preprocessing workers as NPY files.

Downsampling occurs before pose conversion and feature extraction, ensuring
that velocity and contact features match the saved frame rate.

## Motion Format

Each NPY file contains one raw array:

```text
float32[length, 366], where 1 <= length <= num_frames
```

The `motion_jepa_366_v1` feature layout is:

| Slice | Size | Content |
|---|---:|---|
| `0:3` | 3 | Root position |
| `3:5` | 2 | Root heading as cosine and sine |
| `5:92` | 87 | Root-local positions for 29 non-root joints |
| `92:272` | 180 | Global rotations in continuous 6D |
| `272:362` | 90 | Global joint velocities |
| `362:366` | 4 | Foot contacts |

Canonicalization aligns the initial heading to zero and moves the initial root
to the horizontal origin while preserving its vertical position.

## Splits and Metadata

The official train list remains the train split. The content and repetition
test lists are merged and deduplicated, entries overlapping train are removed,
and the remaining IDs are deterministically shuffled with seed 42. One third
becomes validation and the remainder becomes test.

The output layout is:

```text
dataset/bones-seed-processed/
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

Split-file rows have the form:

```text
sample_id,relative_npy_path,fps,actual_length
```

`index.json` stores the same FPS and frame count plus a unique segment ID,
source ID, segment index, exclusive start/end frame bounds, captions, and
selected metadata and NPY paths. Training-only frame statistics are written to
`stats/mean.npy` and `stats/std.npy`; motion arrays remain unnormalized.
The fixed dataset FPS is stored once in `meta.json`.

## Failure Handling

Failures are isolated per motion. Missing or malformed BVHs, invalid FPS,
motions shorter than `min_frames`, and other conversion failures are written
to `errors.jsonl`; other work continues and failed samples are omitted from
the generated splits and statistics.

With the defaults, 120 and 180 FPS become 60 FPS with strides 2 and 3; 30 FPS
is below the configured rate and 90 FPS is not divisible, so both are excluded.

## Storage and Performance

The final output directory is created before conversion, and workers write
each sequence NPY directly into it. Complete output is reused unless
`--overwrite` is supplied. Interrupted output remains visible with a build
marker and requires `--overwrite` to restart.

Multiprocessing uses one PyTorch thread per worker. `--chunksize` controls how
many motions are dispatched to a worker in one scheduling operation; its
default is 16. Chunking reduces multiprocessing communication overhead but
does not combine motions or change outputs.

Each conversion worker saves its own uniquely named NPY clips, while returning
only compact records and statistics to the parent. Standard
`DistributedSampler` provides globally randomized batches and equal per-rank
sample counts. Loader workers open only the files needed by each batch; placing
the processed directory on SSD avoids cold random HDD seek stalls.

Example:

```bash
python dataset/preprocess_dataset.py \
  --workers 64 \
  --chunksize 16 \
  --num_frames 300 \
  --min_frames 90 \
  --fps 60 \
  --overlap 0.5
```

The loader normalizes only the real rows, appends literal zero rows to
`num_frames`, and returns `(motion, fps, valid_length)`. Model masks and target
attention use that valid length, so padding cannot become context or a target.

Other useful arguments are `--max_per_split`, `--split_seed`, and `--overwrite`.

## Training Integration

The default training configuration points to:

```yaml
data:
  root_path: ./dataset/bones-seed-processed
  meta_files:
    - train.txt
  num_frames: 300
  fps: 60
  motion_dim: 366
  normalize: true
  stats_path: stats
```

The loader verifies configured FPS against `meta.json`, length against the
split row, manifest hash, and NPY shape/dtype. It consumes each saved segment whole
without runtime resampling, applies
training statistics only to real frames, and end-pads with zeros. It passes FPS
and valid length to training for time-aware positions and padding exclusion.
