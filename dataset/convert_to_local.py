"""Convert an unnormalized 366D processed dataset to LocalMotionJEPARep.

No BVH parsing or forward kinematics is performed. Source files are read-only.
The destination must be new; meta.json is written last as the completion marker.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch
from tqdm import tqdm

from dataset.motion_dataset import MotionDataset
from motion_rep import LocalMotionJEPARep, MotionJEPAMotionRep


_SOURCE = None
_OUTPUT = None
_REP = None


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def _init_worker(source, output, fps):
    global _SOURCE, _OUTPUT, _REP
    torch.set_num_threads(1)
    _SOURCE, _OUTPUT = Path(source), Path(output)
    _REP = LocalMotionJEPARep(fps=fps)


def _convert_clip(item):
    relative, length, training_weight = item
    try:
        source = (_SOURCE / relative).resolve()
        destination = _OUTPUT / relative
        if not source.is_relative_to(_SOURCE):
            raise ValueError(f"Motion path escapes dataset: {relative}")
        original = np.load(source, allow_pickle=False)
        if original.dtype != np.float32 or original.shape != (length, 366):
            raise ValueError(f"Invalid input dtype/shape: {relative}: {original.dtype}, {original.shape}")
        # Calculate in float64 to avoid adding unnecessary rounding to stored float32 data.
        converted = _REP.from_global_features(torch.from_numpy(original).double())
        motion = np.ascontiguousarray(converted.numpy(), dtype=np.float32)
        if not np.isfinite(motion).all():
            raise ValueError(f"Non-finite converted features: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            np.save(stream, motion, allow_pickle=False)
        if training_weight:
            values = motion.astype(np.float64)
            return (length * training_weight, values.sum(0) * training_weight,
                    np.square(values).sum(0) * training_weight)
        return (0, None, None)
    except Exception:
        raise RuntimeError(f"Failed converting {relative}\n{traceback.format_exc()}") from None


def convert_dataset(source, output, workers=8):
    source, output = Path(source).resolve(), Path(output).resolve()
    if workers < 1:
        raise ValueError("workers must be positive")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate, non-nested directories")
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    metadata = json.loads((source / "meta.json").read_text())
    if metadata.get("representation") != MotionJEPAMotionRep.FORMAT_NAME:
        raise ValueError("Source must use the original motion_jepa_366_v1 representation")
    datasets = {
        split: MotionDataset(source, f"{split}.txt", metadata["num_frames"], metadata["fps"])
        for split in ("train", "val", "test")
    }
    if not len(datasets["train"]):
        raise ValueError("Source has no training clips")
    # A shared motion file can occur multiple times in a labelled index.
    lengths, weights = {}, Counter()
    for split, dataset in datasets.items():
        for entry in dataset.entries:
            relative = entry.path.relative_to(source).as_posix()
            if relative in lengths and lengths[relative] != entry.length:
                raise ValueError(f"Conflicting lengths for {relative}")
            lengths[relative] = entry.length
            if split == "train":
                weights[relative] += 1
    records = json.loads((source / "index.json").read_text())
    for record in records:
        record["motion_dim"] = LocalMotionJEPARep.FEATURE_DIM
    output.mkdir(parents=True)
    count = 0
    total = np.zeros(365, dtype=np.float64)
    total_squared = np.zeros(365, dtype=np.float64)
    tasks = ((path, length, weights[path]) for path, length in lengths.items())
    pool = None
    try:
        if workers == 1:
            _init_worker(source, output, metadata["fps"])
            results = map(_convert_clip, tasks)
        else:
            pool = Pool(workers, initializer=_init_worker, initargs=(source, output, metadata["fps"]))
            results = pool.imap(_convert_clip, tasks, chunksize=16)
        for n, sums, squared in tqdm(results, total=len(lengths), desc="366D -> 365D", unit="clip"):
            if n:
                count += n
                total += sums
                total_squared += squared
        if pool is not None:
            pool.close()
            pool.join()
            pool = None
        mean = total / count
        std = np.sqrt(np.maximum(total_squared / count - mean * mean, 1e-12))
        std = np.where(std < 1e-6, 1.0, std)
        (output / "stats").mkdir()
        np.save(output / "stats/mean.npy", mean.astype(np.float32))
        np.save(output / "stats/std.npy", std.astype(np.float32))
        for split in datasets:
            shutil.copy2(source / f"{split}.txt", output / f"{split}.txt")
            manifest = json.loads((source / f"motions/{split}.json").read_text())
            manifest.update(representation=LocalMotionJEPARep.FORMAT_NAME, motion_dim=365)
            _write_json(output / f"motions/{split}.json", manifest)
        # Preserve annotations and source error records, but never old feature statistics.
        for name in ("errors.jsonl", "class-index.json"):
            if (source / name).is_file():
                shutil.copy2(source / name, output / name)
        if (source / "labels").is_dir():
            shutil.copytree(source / "labels", output / "labels")
        for record in records:
            if "motion_sha256" in record:
                record["motion_sha256"] = _sha256(output / record["motion_path"])
        _write_json(output / "index.json", records)
        metadata.update(
            representation=LocalMotionJEPARep.FORMAT_NAME, motion_dim=365,
            train_stats_frames=count, write_mode="converted_from_processed_366d",
            statistics={"format": "numpy_float32_pair", "mean_path": "stats/mean.npy",
                        "std_path": "stats/std.npy", "source_split": "train", "valid_frames_only": True},
            representation_conversion={
                "version": 1, "source": str(source),
                "source_representation": MotionJEPAMotionRep.FORMAT_NAME,
                "source_meta_sha256": _sha256(source / "meta.json"),
                "source_index_sha256": _sha256(source / "index.json"),
                "method": "rotate_stored_body_features_and_velocities_to_current_heading",
            },
        )
        if "index_sha256" in metadata:
            metadata["index_sha256"] = _sha256(output / "index.json")
        _write_json(output / "meta.json", metadata)
        for split in datasets:
            MotionDataset(output, f"{split}.txt", metadata["num_frames"], metadata["fps"],
                          normalize=True, representation="LocalMotionJEPARep")
    except Exception:
        (output / "conversion-error.txt").write_text(traceback.format_exc())
        # Never advertise an incomplete/invalid output as a completed dataset.
        (output / "meta.json").unlink(missing_ok=True)
        raise
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
    print(f"Converted {len(lengths)} unique clips; training statistics: {count} frames.")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original unnormalized 366D dataset")
    parser.add_argument("--output", type=Path, required=True, help="New local-representation directory")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    convert_dataset(args.input, args.output, args.workers)


if __name__ == "__main__":
    main()
