"""Preprocess 100STYLE SOMA77 BVHs into shuffled, non-overlapping NPY windows."""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import traceback
from dataclasses import dataclass, replace
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.preprocess_dataset import (  # noqa: E402
    SPLITS,
    _prepare_output,
    _save_record_motion,
    _validate_complete_dataset,
    finalize_processed_dataset,
    resample_motion_fps,
    round_fps,
)
from motion_rep import MotionJEPAMotionRep  # noqa: E402
from skeleton import SOMASkeleton30, parse_bvh_motion  # noqa: E402


SOURCE_PATTERN = "bvh/*_soma77.bvh"
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

_DATASET_ROOT: Path | None = None
_OUTPUT_ROOT: Path | None = None
_FPS = 30
_TARGET_SKELETON: SOMASkeleton30 | None = None
_THREAD_CONFIG_PID: int | None = None


@dataclass(frozen=True)
class SourceMotion:
    id: str
    bvh_path: str
    style: str
    motion_code: str


@dataclass(frozen=True)
class WindowDescriptor:
    source_id: str
    segment_index: int
    start_frame: int
    end_frame: int
    split: str = ""


@dataclass(frozen=True)
class PlannedSource:
    source: SourceMotion
    source_fps: int
    resampled_frames: int
    windows: tuple[WindowDescriptor, ...]


def parse_style_motion(path: str | Path) -> tuple[str, str, str]:
    """Return the sample id, style label, and motion code from a SOMA77 filename."""
    name = Path(path).name
    suffix = "_soma77.bvh"
    if not name.endswith(suffix):
        raise ValueError(f"100STYLE input must end with {suffix!r}: {name}")
    sample_id = name[: -len(suffix)]
    if "_" not in sample_id:
        raise ValueError(f"100STYLE filename has no motion code: {name}")
    style, motion_code = sample_id.rsplit("_", 1)
    if not style or not motion_code:
        raise ValueError(f"Invalid 100STYLE filename: {name}")
    return sample_id, style, motion_code


def discover_sources(dataset_root: Path, limit: int = -1) -> list[SourceMotion]:
    """Discover SOMA77 source files in stable filename order."""
    paths = sorted(dataset_root.glob(SOURCE_PATTERN))
    if limit >= 0:
        paths = paths[:limit]
    sources = []
    for path in paths:
        sample_id, style, motion_code = parse_style_motion(path)
        sources.append(
            SourceMotion(
                id=sample_id,
                bvh_path=path.relative_to(dataset_root).as_posix(),
                style=style,
                motion_code=motion_code,
            )
        )
    if not sources:
        raise FileNotFoundError(
            f"No 100STYLE SOMA77 BVHs found with pattern {SOURCE_PATTERN!r} "
            f"under {dataset_root}"
        )
    return sources


def read_bvh_timing(path: Path) -> tuple[int, int]:
    """Read source frame count and rounded FPS without loading BVH frame data."""
    frame_count: int | None = None
    frame_time: float | None = None
    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if line.startswith("Frames:"):
                frame_count = int(line.split(":", 1)[1])
            elif line.startswith("Frame Time:"):
                frame_time = float(line.split(":", 1)[1])
                break
    if frame_count is None or frame_time is None or frame_count <= 0 or frame_time <= 0:
        raise ValueError(f"Invalid BVH motion timing header: {path}")
    return frame_count, round_fps(1.0 / frame_time)


def resampled_frame_count(frame_count: int, source_fps: int, target_fps: int) -> int:
    """Return the length produced by frame-zero-aligned fixed-step indexing."""
    if source_fps < target_fps:
        raise ValueError(
            f"Rounded BVH frame rate is below configured FPS: {source_fps} < {target_fps}"
        )
    if source_fps % target_fps != 0:
        raise ValueError(
            "Rounded BVH frame rate must be an exact multiple of configured "
            f"FPS {target_fps}, got {source_fps}"
        )
    step = source_fps // target_fps
    return int(math.ceil(frame_count / step))


def enumerate_windows(
    source_id: str, frame_count: int, num_frames: int
) -> list[WindowDescriptor]:
    """Enumerate complete, non-overlapping windows and discard the final remainder."""
    return [
        WindowDescriptor(source_id, index, start, start + num_frames)
        for index, start in enumerate(range(0, frame_count - num_frames + 1, num_frames))
    ]


def assign_window_splits(
    windows: list[WindowDescriptor], seed: int
) -> list[WindowDescriptor]:
    """Shuffle all windows and assign the closest integer 80/10/10 split."""
    shuffled = list(windows)
    random.Random(seed).shuffle(shuffled)
    quotas = {split: len(shuffled) * SPLIT_RATIOS[split] for split in SPLITS}
    counts = {split: int(math.floor(quotas[split])) for split in SPLITS}
    remaining = len(shuffled) - sum(counts.values())
    remainder_order = sorted(
        SPLITS,
        key=lambda split: (-(quotas[split] - counts[split]), SPLITS.index(split)),
    )
    for split in remainder_order[:remaining]:
        counts[split] += 1
    num_train = counts["train"]
    num_val = counts["val"]
    assigned = []
    for index, window in enumerate(shuffled):
        if index < num_train:
            split = "train"
        elif index < num_train + num_val:
            split = "val"
        else:
            split = "test"
        assigned.append(replace(window, split=split))
    return assigned


def build_window_plan(
    sources: list[SourceMotion],
    dataset_root: Path,
    num_frames: int,
    fps: int,
    seed: int,
) -> tuple[list[PlannedSource], list[dict[str, Any]]]:
    """Inspect source headers, enumerate windows, and assign global window splits."""
    source_info: dict[str, tuple[int, int]] = {}
    all_windows: list[WindowDescriptor] = []
    errors: list[dict[str, Any]] = []
    for source in sources:
        path = Path(source.bvh_path)
        try:
            raw_frames, source_fps = read_bvh_timing(
                _absolute_source_path(source, dataset_root)
            )
            frame_count = resampled_frame_count(raw_frames, source_fps, fps)
            windows = enumerate_windows(source.id, frame_count, num_frames)
            if not windows:
                raise ValueError(
                    f"Motion has {frame_count} resampled frames, fewer than {num_frames}"
                )
            source_info[source.id] = (source_fps, frame_count)
            all_windows.extend(windows)
        except Exception as error:
            errors.append(
                {
                    "ok": False,
                    "id": source.id,
                    "split": "unassigned",
                    "source_path": path.as_posix(),
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(limit=8),
                }
            )

    assigned = assign_window_splits(all_windows, seed)
    by_source: dict[str, list[WindowDescriptor]] = {source.id: [] for source in sources}
    for window in assigned:
        by_source[window.source_id].append(window)
    planned = []
    for source in sources:
        if source.id not in source_info:
            continue
        source_fps, frame_count = source_info[source.id]
        windows = sorted(by_source[source.id], key=lambda window: window.segment_index)
        planned.append(PlannedSource(source, source_fps, frame_count, tuple(windows)))
    return planned, errors


def _absolute_source_path(source: SourceMotion, dataset_root: Path | None) -> Path:
    root = dataset_root if dataset_root is not None else _DATASET_ROOT
    if root is None:
        raise RuntimeError("100STYLE dataset root is not initialized")
    return root / source.bvh_path


def _init_worker(dataset_root: str, output_root: str, fps: int) -> None:
    global _DATASET_ROOT, _OUTPUT_ROOT, _FPS
    global _TARGET_SKELETON, _THREAD_CONFIG_PID
    _DATASET_ROOT = Path(dataset_root)
    _OUTPUT_ROOT = Path(output_root)
    _FPS = int(fps)
    current_pid = os.getpid()
    if _THREAD_CONFIG_PID != current_pid:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        _THREAD_CONFIG_PID = current_pid
    _TARGET_SKELETON = SOMASkeleton30()


def _convert_source(item: PlannedSource) -> dict[str, Any]:
    assert _DATASET_ROOT is not None and _OUTPUT_ROOT is not None
    assert _TARGET_SKELETON is not None
    source = item.source
    source_path = _absolute_source_path(source, _DATASET_ROOT)
    try:
        local_rotations, root_positions, parsed_fps = parse_bvh_motion(source_path)
        source_fps = round_fps(float(parsed_fps))
        if source_fps != item.source_fps:
            raise ValueError(
                f"BVH FPS changed after planning: {source_fps} != {item.source_fps}"
            )
        local_rotations, root_positions, fps = resample_motion_fps(
            local_rotations, root_positions, source_fps, _FPS
        )
        if len(local_rotations) != item.resampled_frames:
            raise ValueError(
                "BVH frame count changed after planning: "
                f"{len(local_rotations)} != {item.resampled_frames}"
            )
        # 100STYLE_soma77 is already expressed on the standard SOMA77 skeleton.
        # Applying the SOMA77 rest-pose offsets again would double-transform it.
        local_rotations = _TARGET_SKELETON.from_soma77(local_rotations)
        motion_rep = MotionJEPAMotionRep(_TARGET_SKELETON, fps)
        records = []
        for window in item.windows:
            features = motion_rep(
                local_rotations[window.start_frame : window.end_frame],
                root_positions[window.start_frame : window.end_frame],
                to_canonicalize=True,
            )
            motion = np.ascontiguousarray(features.detach().cpu().numpy(), dtype=np.float32)
            expected = (window.end_frame - window.start_frame, MotionJEPAMotionRep.FEATURE_DIM)
            if motion.shape != expected:
                raise ValueError(f"Unexpected feature shape {motion.shape}, expected {expected}")
            if not np.isfinite(motion).all():
                raise ValueError("Converted feature array contains non-finite values")
            sample_id = f"{source.id}_{window.segment_index:04d}"
            metadata = {
                "style": source.style,
                "motion_code": source.motion_code,
                "source_id": source.id,
                "start_frame": window.start_frame,
                "end_frame": window.end_frame,
            }
            record = {
                "id": sample_id,
                "source_id": source.id,
                "segment_index": window.segment_index,
                "start_frame": window.start_frame,
                "end_frame": window.end_frame,
                "split": window.split,
                "source_path": source.bvh_path,
                "source_fps": source_fps,
                "fps": fps,
                "length": int(len(motion)),
                "motion_dim": int(motion.shape[1]),
                "metadata": metadata,
                "motion": motion,
            }
            records.append(_save_record_motion(record, _OUTPUT_ROOT))
        return {"ok": True, "id": source.id, "split": "mixed", "records": records}
    except Exception as error:
        return {
            "ok": False,
            "id": source.id,
            "split": "mixed",
            "source_path": source.bvh_path,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(limit=8),
        }


def _convert_batch(items: list[PlannedSource]) -> list[dict[str, Any]]:
    return [_convert_source(item) for item in items]


def _ordered_results(
    items: list[PlannedSource], args: argparse.Namespace
) -> Iterator[dict[str, Any]]:
    if not items:
        return
    init_args = (str(args.dataset_root), str(args.output), args.fps)
    worker_count = min(max(1, int(args.workers)), len(items))
    if worker_count == 1:
        _init_worker(*init_args)
        yield from (_convert_source(item) for item in items)
        return
    chunk_size = max(1, int(args.chunksize))
    batches = [items[start : start + chunk_size] for start in range(0, len(items), chunk_size)]
    with Pool(worker_count, initializer=_init_worker, initargs=init_args) as pool:
        pending = []
        next_batch = 0
        while next_batch < len(batches) and len(pending) < worker_count * 2:
            pending.append(pool.apply_async(_convert_batch, (batches[next_batch],)))
            next_batch += 1
        while pending:
            yield from pending.pop(0).get()
            if next_batch < len(batches):
                pending.append(pool.apply_async(_convert_batch, (batches[next_batch],)))
                next_batch += 1


def _validate_config(args: argparse.Namespace) -> None:
    if args.num_frames <= 0:
        raise ValueError(f"--num_frames must be positive, got {args.num_frames}")
    if args.fps <= 0:
        raise ValueError(f"--fps must be positive, got {args.fps}")
    if args.limit == 0:
        raise ValueError("--limit must be positive or negative for no limit")


def preprocess(args: argparse.Namespace) -> None:
    """Discover, split, convert, and finalize the 100STYLE NPY dataset."""
    _validate_config(args)
    args.dataset_root = Path(args.dataset_root).resolve()
    args.output = Path(args.output).resolve()
    args.overlap = 0.0
    if args.output.exists() and _validate_complete_dataset(args.output) and not args.overwrite:
        print(f"Reusing complete NPY dataset: {args.output}")
        return

    sources = discover_sources(args.dataset_root, args.limit)
    planned, errors = build_window_plan(
        sources,
        args.dataset_root,
        args.num_frames,
        args.fps,
        args.split_seed,
    )
    if not planned:
        raise RuntimeError("No complete 100STYLE windows were discovered")

    _prepare_output(args.output, args.overwrite)
    records_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    successful_sources: set[str] = set()
    sources_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    for result in tqdm(
        _ordered_results(planned, args),
        total=len(planned),
        desc="100STYLE BVH -> NPY",
        unit="motion",
    ):
        if not result["ok"]:
            errors.append(result)
            continue
        successful_sources.add(result["id"])
        for record in result["records"]:
            if "motion" in record:
                _save_record_motion(record, args.output)
            split = record["split"]
            records_by_split[split].append(record)
            sources_by_split[split].add(result["id"])

    finalize_processed_dataset(
        args,
        records_by_split,
        errors,
        {split: len(sources_by_split[split]) for split in SPLITS},
        source_dataset="100STYLE_soma77",
        segmentation="complete_non_overlapping_windows_shuffled_globally",
        num_source_motions=len(successful_sources),
        metadata_extra={
            "split_unit": "window",
            "split_ratios": SPLIT_RATIOS,
            "source_pattern": SOURCE_PATTERN,
            "tail_policy": "discard_incomplete_window",
            "source_standard_tpose": True,
            "num_discovered_source_motions": len(sources),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert 100STYLE SOMA77 BVHs to shuffled Motion-JEPA NPY windows."
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=PROJECT_ROOT / "dataset/100STYLE_soma77",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "dataset/100style-processed",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, os.cpu_count() or 1),
        help="Worker processes. Each worker uses one PyTorch CPU thread.",
    )
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=90)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        metavar="N",
        help="Process at most N source BVHs; negative values process every source.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    preprocess(parse_args())
