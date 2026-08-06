"""Small NPY-backed dataset fixtures shared by integration tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def write_npy_dataset(
    root: Path,
    motions: list[np.ndarray],
    *,
    split: str = "train",
    num_frames: int | None = None,
    fps: int = 60,
) -> None:
    if not motions:
        raise ValueError("motions cannot be empty")
    arrays = [np.ascontiguousarray(motion, dtype=np.float32) for motion in motions]
    motion_dim = arrays[0].shape[1]
    num_frames = int(num_frames or max(len(motion) for motion in arrays))
    motion_root = root / "motions" / split
    motion_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, motion in enumerate(arrays):
        relative_path = Path("motions") / split / f"sample-{index}.npy"
        np.save(root / relative_path, motion, allow_pickle=False)
        rows.append(f"sample-{index},{relative_path.as_posix()},{fps},{len(motion)}\n")
    index_path = root / f"{split}.txt"
    index_path.write_text("".join(rows), encoding="utf-8")
    manifest = {
        "format": "motion_jepa_npy_v1",
        "format_version": 1,
        "split": split,
        "motion_root": f"motions/{split}",
        "num_samples": len(arrays),
        "dtype": "float32",
        "representation": "motion_jepa_366_v1",
        "fps": fps,
        "num_frames": num_frames,
        "motion_dim": motion_dim,
        "split_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
    }
    (root / "motions" / f"{split}.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    (root / "meta.json").write_text(
        json.dumps(
            {
                "representation": "motion_jepa_366_v1",
                "motion_storage": "npy_float32_v1",
                "motion_dim": motion_dim,
                "num_frames": num_frames,
                "fps": fps,
            }
        )
        + "\n",
        encoding="utf-8",
    )
