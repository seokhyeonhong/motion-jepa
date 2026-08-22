"""Shared 100STYLE dataset and deterministic style-label indexing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from dataset import MotionDataset


DEFAULT_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class StyleLabelIndex:
    """Stable mappings between style names, class IDs, and sample IDs."""

    class_names: tuple[str, ...]
    class_to_index: dict[str, int]
    style_by_id: dict[str, str]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def label_for_sample(self, sample_id: str) -> int:
        try:
            style = self.style_by_id[sample_id]
        except KeyError as error:
            raise KeyError(f"Sample ID is missing from the style index: {sample_id}") from error
        return self.class_to_index[style]

    def to_json(self) -> dict[str, object]:
        return {
            "class_names": list(self.class_names),
            "class_to_index": dict(self.class_to_index),
        }


def load_style_label_index(dataset_root: str | Path) -> StyleLabelIndex:
    """Read index.json and assign class IDs by sorted style name."""
    path = Path(dataset_root) / "index.json"
    if not path.is_file():
        raise FileNotFoundError(f"100STYLE index does not exist: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"100STYLE index is empty or malformed: {path}")

    style_by_id: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Index record must be an object: {record!r}")
        sample_id = record.get("id")
        metadata = record.get("metadata")
        style = metadata.get("style") if isinstance(metadata, dict) else None
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Index record has no valid sample ID: {record}")
        if not isinstance(style, str) or not style:
            raise ValueError(f"Index record has no valid style label: {record}")
        if sample_id in style_by_id:
            raise ValueError(f"Duplicate sample ID in index: {sample_id}")
        style_by_id[sample_id] = style

    class_names = tuple(sorted(set(style_by_id.values())))
    class_to_index = {name: index for index, name in enumerate(class_names)}
    return StyleLabelIndex(class_names, class_to_index, style_by_id)


def load_style_index(dataset_root: str | Path) -> tuple[list[str], dict[str, str]]:
    """Compatibility tuple for callers predating StyleLabelIndex."""
    index = load_style_label_index(dataset_root)
    return list(index.class_names), dict(index.style_by_id)


class StyleMotionDataset(Dataset):
    """Attach deterministic style labels and sample IDs to MotionDataset."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        num_frames: int,
        fps: int,
        motion_dim: int,
        stats_root: str | Path,
        label_index: StyleLabelIndex,
    ) -> None:
        self.motion = MotionDataset(
            root_path=root,
            meta_files=f"{split}.txt",
            num_frames=num_frames,
            fps=fps,
            motion_dim=motion_dim,
            normalize=True,
            stats_path=stats_root,
        )
        self.label_index = label_index
        self.sample_ids = [entry.sample_id for entry in self.motion.entries]
        missing = [
            sample_id
            for sample_id in self.sample_ids
            if sample_id not in label_index.style_by_id
        ]
        if missing:
            raise ValueError(
                f"Split {split!r} has sample IDs missing from index.json: {missing[:3]}"
            )
        self.labels = [label_index.label_for_sample(sample_id) for sample_id in self.sample_ids]

    def __len__(self) -> int:
        return len(self.motion)

    def __getitem__(self, index: int):
        motion, fps, length = self.motion[index]
        return motion, fps, length, self.labels[index], self.sample_ids[index]


class StyleTokenDataset(Dataset):
    """Expose a validated cached JEPA token split through the motion batch API."""

    def __init__(
        self,
        payload: dict[str, object],
        *,
        label_index: StyleLabelIndex,
        fps: int,
    ) -> None:
        features = payload.get("features")
        lengths = payload.get("lengths")
        labels = payload.get("labels")
        sample_ids = payload.get("sample_ids")
        if not isinstance(features, torch.Tensor) or features.ndim != 3:
            raise ValueError("Token cache features must have shape [N,T,D]")
        if features.dtype != torch.bfloat16:
            raise ValueError("Token cache features must use bfloat16")
        if not isinstance(lengths, torch.Tensor) or lengths.dtype != torch.long:
            raise ValueError("Token cache lengths must be int64")
        if not isinstance(labels, torch.Tensor) or labels.dtype != torch.long:
            raise ValueError("Token cache labels must be int64")
        if not isinstance(sample_ids, list) or not all(
            isinstance(sample_id, str) for sample_id in sample_ids
        ):
            raise ValueError("Token cache sample_ids must be a list of strings")
        count = len(features)
        if len(lengths) != count or len(labels) != count or len(sample_ids) != count:
            raise ValueError("Token cache fields have inconsistent sample counts")
        if (lengths < 0).any() or (lengths > features.shape[1]).any():
            raise ValueError("Token cache contains invalid sequence lengths")
        expected_labels = torch.tensor(
            [label_index.label_for_sample(sample_id) for sample_id in sample_ids],
            dtype=torch.long,
        )
        if not torch.equal(labels, expected_labels):
            raise ValueError("Token cache labels do not match the style index")
        if not torch.isfinite(features).all():
            raise ValueError("Token cache contains non-finite features")
        self.features = features
        self.lengths = lengths
        self.labels = labels
        self.sample_ids = sample_ids
        self.fps = int(fps)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return (
            self.features[index],
            self.fps,
            self.lengths[index],
            self.labels[index],
            self.sample_ids[index],
        )


def build_style_datasets(
    root: str | Path,
    *,
    splits: Iterable[str] = DEFAULT_SPLITS,
    num_frames: int,
    fps: int,
    motion_dim: int,
    stats_root: str | Path,
    label_index: StyleLabelIndex | None = None,
) -> tuple[dict[str, StyleMotionDataset], StyleLabelIndex]:
    """Build split datasets that all share one stable style-label index."""
    root = Path(root)
    resolved_index = label_index or load_style_label_index(root)
    datasets = {
        split: StyleMotionDataset(
            root,
            split,
            num_frames=num_frames,
            fps=fps,
            motion_dim=motion_dim,
            stats_root=stats_root,
            label_index=resolved_index,
        )
        for split in splits
    }
    return datasets, resolved_index


__all__ = [
    "DEFAULT_SPLITS",
    "StyleLabelIndex",
    "StyleMotionDataset",
    "StyleTokenDataset",
    "build_style_datasets",
    "load_style_index",
    "load_style_label_index",
]
