"""Fixed training subsets with at least one supervised window per category."""

import hashlib
import json
import math

import numpy as np
import torch
from torch.utils.data import Subset


class SegmentationSubset(Subset):
    """Expose the metadata used by the frame trainer and RAM feature extractor."""

    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.split = dataset.split
        self.num_classes = dataset.num_classes
        self.class_names = dataset.class_names
        self.class_indices_60 = dataset.class_indices_60
        self.num_frames = dataset.num_frames
        self.motion_dim = dataset.motion_dim


def select_training_subset(dataset, fraction=1.0, seed=42):
    """Build a fraction-independent coverage core, then fill from a permutation.

    Only label files are read. Fractions with the same seed are nested, and
    returned rows retain dataset order. Full-data runs keep their existing path.
    """
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("train_fraction must be finite and in (0, 1]")
    if dataset.split != "train":
        raise ValueError("Subset selection is only supported for the training split")
    if fraction == 1:
        return dataset, None
    count = math.floor(len(dataset) * fraction)
    if count < 1:
        raise ValueError("train_fraction selects zero training windows")
    positive_frames = np.zeros((len(dataset), dataset.num_classes), dtype=np.int64)
    for index in range(len(dataset)):
        labels, supervision = dataset.load_labels(index)
        positive_frames[index] = labels[supervision].sum(axis=0)
    present = positive_frames > 0
    frequencies = present.sum(axis=0)
    missing = [dataset.class_names[i] for i in np.flatnonzero(frequencies == 0)]
    if missing:
        raise ValueError(f"Required categories absent from the training split: {missing}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    core = []
    covered = np.zeros(dataset.num_classes, dtype=np.bool_)
    for category in np.argsort(frequencies, kind="stable"):
        if covered[category]:
            continue
        candidates = np.flatnonzero(present[:, category])
        index = int(candidates[int(torch.randint(len(candidates), (), generator=generator))])
        core.append(index)
        covered |= present[index]
    if len(core) > count:
        raise ValueError(
            f"train_fraction={fraction} selects {count} windows, but this seed's label coverage "
            f"core needs {len(core)}; increase the fraction (the budget is not enlarged automatically)"
        )
    chosen = set(core)
    remaining = [i for i in torch.randperm(len(dataset), generator=generator).tolist() if i not in chosen]
    indices = sorted(core + remaining[:count - len(core)])
    sample_ids = [dataset.records[i]["id"] for i in indices]
    metadata = {
        "fraction": float(fraction), "seed": int(seed),
        "available_samples": len(dataset), "selected_samples": len(indices),
        "actual_fraction": len(indices) / len(dataset), "coverage_core_samples": len(core),
        "policy": "rarest_first_random_cover_permutation_floor_v1",
        "sample_ids": sample_ids,
        "sample_ids_sha256": hashlib.sha256(json.dumps(sample_ids, separators=(",", ":")).encode()).hexdigest(),
        "class_names": list(dataset.class_names),
        "available_positive_windows": frequencies.tolist(),
        "selected_positive_windows": present[indices].sum(axis=0).tolist(),
        "selected_positive_frames": positive_frames[indices].sum(axis=0).tolist(),
    }
    return SegmentationSubset(dataset, indices), metadata
