"""Preprocess frame-cut 100STYLE SOMA77 BVHs into shuffled variable-length NPY windows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
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

from dataset.preprocess_bones_seed import (  # noqa: E402
    SPLITS,
    _prepare_output,
    _protocol_sha256,
    _save_record_motion,
    _validate_complete_dataset,
    finalize_processed_dataset,
    resample_motion_fps,
    round_fps,
)
from motion_rep.registry import (get_representation, add_representation_argument,
                                 validate_representation_output)  # noqa: E402

_REPRESENTATION = get_representation()
from skeleton import SOMASkeleton30, parse_bvh_motion  # noqa: E402


SOURCE_PATTERN = "bvh/*_soma77.bvh"
DEFAULT_CONTENTS = ("BR", "BW", "FR", "FW", "SR", "SW")
KNOWN_CONTENTS = ("BR", "BW", "FR", "FW", "ID", "SR", "SW", "TR1", "TR2", "TR3")
PREPROCESSING_VERSION = "100style_window_split"
SEGMENTATION_POLICY = "frame_cut_non_overlapping_windows_with_short_tail"
SPLIT_PROPORTIONS = {"train": 0.85, "val": 0.05, "test": 0.10}
FRAME_CUT_STOP_CONVENTION = "exclusive"

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
    source_frames: int
    cut_start_frame: int
    cut_stop_frame: int
    resampled_frames: int
    windows: tuple[WindowDescriptor, ...]


@dataclass(frozen=True)
class FrameCut:
    start_frame: int
    stop_frame: int


def load_frame_cuts(path: str | Path) -> dict[str, FrameCut]:
    """Load strict source-frame cuts keyed by ``<style>_<content>``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"100STYLE frame-cut CSV does not exist: {path}")
    required = {"STYLE_NAME"}
    for content in KNOWN_CONTENTS:
        required.update((f"{content}_START", f"{content}_STOP"))
    cuts: dict[str, FrameCut] = {}
    seen_styles: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing_columns = sorted(required - set(reader.fieldnames or ()))
        if missing_columns:
            raise ValueError(
                f"Frame-cut CSV is missing columns {missing_columns}: {path}"
            )
        for line_number, row in enumerate(reader, start=2):
            style = str(row.get("STYLE_NAME", "")).strip()
            if not style:
                raise ValueError(f"Frame-cut CSV has an empty style at line {line_number}")
            if style in seen_styles:
                raise ValueError(f"Duplicate style in frame-cut CSV: {style}")
            seen_styles.add(style)
            for content in KNOWN_CONTENTS:
                raw_start = str(row.get(f"{content}_START", "")).strip()
                raw_stop = str(row.get(f"{content}_STOP", "")).strip()
                if raw_start == "N/A" and raw_stop == "N/A":
                    continue
                if "N/A" in (raw_start, raw_stop) or not raw_start or not raw_stop:
                    raise ValueError(
                        f"Incomplete {style}_{content} frame cut at line {line_number}"
                    )
                try:
                    start, stop = int(raw_start), int(raw_stop)
                except ValueError as error:
                    raise ValueError(
                        f"Non-integer {style}_{content} frame cut at line {line_number}"
                    ) from error
                if start < 0 or stop <= start:
                    raise ValueError(
                        f"Invalid {style}_{content} frame cut [{start}, {stop})"
                    )
                cuts[f"{style}_{content}"] = FrameCut(start, stop)
    if not cuts:
        raise ValueError(f"Frame-cut CSV contains no cuts: {path}")
    return cuts


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


def discover_sources(
    dataset_root: Path,
    limit: int = -1,
    contents: tuple[str, ...] | None = None,
) -> list[SourceMotion]:
    """Discover SOMA77 source files in stable filename order."""
    paths = sorted(dataset_root.glob(SOURCE_PATTERN))
    sources = []
    for path in paths:
        sample_id, style, motion_code = parse_style_motion(path)
        source = SourceMotion(
            id=sample_id,
            bvh_path=path.relative_to(dataset_root).as_posix(),
            style=style,
            motion_code=motion_code,
        )
        if contents is None or source.motion_code in contents:
            sources.append(source)
    if limit >= 0:
        sources = sources[:limit]
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
    source_id: str, frame_count: int, num_frames: int, min_frames: int
) -> list[WindowDescriptor]:
    """Keep non-overlapping windows, including a remainder of at least min_frames."""
    if not 1 <= min_frames <= num_frames:
        raise ValueError("Expected 1 <= min_frames <= num_frames")
    return [
        WindowDescriptor(source_id, index, start, min(start + num_frames, frame_count))
        for index, start in enumerate(range(0, frame_count, num_frames))
        if frame_count - start >= min_frames
    ]


def assign_window_splits(
    planned: list[PlannedSource], split_seed: int
) -> list[PlannedSource]:
    """Shuffle all windows; floor train/val counts and assign the rest to test."""
    windows = [window for item in planned for window in item.windows]
    order = np.random.default_rng(split_seed).permutation(len(windows))
    train_end = len(windows) * 85 // 100
    val_end = train_end + len(windows) * 5 // 100
    assignments = {}
    for rank, index in enumerate(order):
        window = windows[index]
        split = "train" if rank < train_end else "val" if rank < val_end else "test"
        assignments[(window.source_id, window.segment_index)] = split
    return [
        replace(item, windows=tuple(
            replace(window, split=assignments[(window.source_id, window.segment_index)])
            for window in item.windows
        ))
        for item in planned
    ]


def build_window_plan(
    sources: list[SourceMotion],
    dataset_root: Path,
    num_frames: int,
    fps: int,
    frame_cuts: dict[str, FrameCut],
    min_frames: int,
    split_seed: int,
) -> tuple[list[PlannedSource], list[dict[str, Any]]]:
    """Plan trimmed windows, then randomly split the global window pool."""
    planned: list[PlannedSource] = []
    errors: list[dict[str, Any]] = []
    for source in sources:
        path = Path(source.bvh_path)
        try:
            if source.id not in frame_cuts:
                raise ValueError(f"Frame-cut CSV has no entry for {source.id}")
            raw_frames, source_fps = read_bvh_timing(
                _absolute_source_path(source, dataset_root)
            )
            frame_cut = frame_cuts[source.id]
            if frame_cut.stop_frame > raw_frames:
                raise ValueError(
                    f"Frame cut [{frame_cut.start_frame}, {frame_cut.stop_frame}) "
                    f"exceeds {raw_frames} source frames"
                )
            frame_count = resampled_frame_count(
                frame_cut.stop_frame - frame_cut.start_frame, source_fps, fps
            )
            windows = tuple(enumerate_windows(source.id, frame_count, num_frames, min_frames))
            if not windows:
                continue
            planned.append(
                PlannedSource(
                    source=source,
                    source_fps=source_fps,
                    source_frames=raw_frames,
                    cut_start_frame=frame_cut.start_frame,
                    cut_stop_frame=frame_cut.stop_frame,
                    resampled_frames=frame_count,
                    windows=windows,
                )
            )
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

    return assign_window_splits(planned, split_seed), errors


def _absolute_source_path(source: SourceMotion, dataset_root: Path | None) -> Path:
    root = dataset_root if dataset_root is not None else _DATASET_ROOT
    if root is None:
        raise RuntimeError("100STYLE dataset root is not initialized")
    return root / source.bvh_path


def _init_worker(dataset_root: str, output_root: str, fps: int, representation: str = "MotionJEPAMotionRep") -> None:
    global _REPRESENTATION
    _REPRESENTATION = get_representation(representation)
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
        if len(local_rotations) != item.source_frames:
            raise ValueError(
                "BVH frame count changed after planning: "
                f"{len(local_rotations)} != {item.source_frames}"
            )
        local_rotations = local_rotations[item.cut_start_frame : item.cut_stop_frame]
        root_positions = root_positions[item.cut_start_frame : item.cut_stop_frame]
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
        motion_rep = _REPRESENTATION(_TARGET_SKELETON, fps)
        records = []
        for window in item.windows:
            features = motion_rep(
                local_rotations[window.start_frame : window.end_frame],
                root_positions[window.start_frame : window.end_frame],
                to_canonicalize=True,
            )
            motion = np.ascontiguousarray(features.detach().cpu().numpy(), dtype=np.float32)
            expected = (window.end_frame - window.start_frame, _REPRESENTATION.FEATURE_DIM)
            if motion.shape != expected:
                raise ValueError(f"Unexpected feature shape {motion.shape}, expected {expected}")
            if not np.isfinite(motion).all():
                raise ValueError("Converted feature array contains non-finite values")
            sample_id = f"{source.id}_{window.segment_index:04d}"
            source_step = source_fps // fps
            source_start_frame = item.cut_start_frame + window.start_frame * source_step
            source_stop_frame = min(
                item.cut_stop_frame,
                item.cut_start_frame + window.end_frame * source_step,
            )
            metadata = {
                "style": source.style,
                "motion_code": source.motion_code,
                "source_id": source.id,
                "start_frame": window.start_frame,
                "end_frame": window.end_frame,
                "frame_cut_start": item.cut_start_frame,
                "frame_cut_stop": item.cut_stop_frame,
                "source_start_frame": source_start_frame,
                "source_stop_frame": source_stop_frame,
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
    init_args = (str(args.dataset_root), str(args.output), args.fps, getattr(args, "representation", "MotionJEPAMotionRep"))
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
    if not 1 <= args.min_frames <= args.num_frames:
        raise ValueError("--min_frames must be between 1 and --num_frames")
    if args.split_seed < 0:
        raise ValueError("--split_seed must be non-negative")
    if args.fps <= 0:
        raise ValueError(f"--fps must be positive, got {args.fps}")
    if args.limit == 0:
        raise ValueError("--limit must be positive or negative for no limit")
    contents = tuple(str(value).strip() for value in args.contents)
    if not contents or any(not value for value in contents):
        raise ValueError("--contents must contain at least one non-empty content")
    if len(set(contents)) != len(contents):
        raise ValueError("--contents must not contain duplicates")
    unknown = sorted(set(contents) - set(KNOWN_CONTENTS))
    if unknown:
        raise ValueError(f"Unknown 100STYLE contents: {unknown}")
    args.contents = contents


def preprocess(args: argparse.Namespace) -> None:
    """Discover, split, convert, and finalize the 100STYLE NPY dataset."""
    validate_representation_output(args)
    _validate_config(args)
    args.dataset_root = Path(args.dataset_root).resolve()
    args.frame_cuts = (
        Path(args.frame_cuts) if args.frame_cuts is not None
        else args.dataset_root / "Frame_Cuts.csv"
    ).resolve()
    args.output = Path(args.output).resolve()
    args.overlap = 0.0
    frame_cut_sha256 = hashlib.sha256(args.frame_cuts.read_bytes()).hexdigest()
    protocol = {
        "version": PREPROCESSING_VERSION,
        "segmentation": SEGMENTATION_POLICY,
        "dataset_root": str(args.dataset_root),
        "fps": args.fps,
        "num_frames": args.num_frames,
        "min_frames": args.min_frames,
        "overlap": 0.0,
        "stride_frames": args.num_frames,
        "split_seed": args.split_seed,
        "split_proportions": SPLIT_PROPORTIONS,
        "split_rounding": "floor_train_and_val_remainder_test",
        "limit": args.limit,
        "contents": list(args.contents),
        "validation_enabled": True,
        "frame_cuts_sha256": frame_cut_sha256,
        "frame_cut_stop_convention": FRAME_CUT_STOP_CONVENTION,
        "frame_cut_interval": "start_inclusive_stop_exclusive",
        "frame_cuts_applied_before_resampling": True,
    }
    if args.output.exists() and not args.overwrite and _validate_complete_dataset(
        args.output, preprocessing_version=PREPROCESSING_VERSION,
        segmentation=SEGMENTATION_POLICY,
    ):
        metadata = json.loads((args.output / "meta.json").read_text(encoding="utf-8"))
        if metadata.get("preprocessing_protocol") != protocol:
            raise ValueError(
                "Existing dataset preprocessing protocol does not match the requested "
                f"configuration or inputs: {args.output}; use --overwrite or a new --output"
            )
        print(f"Reusing complete NPY dataset: {args.output}")
        return

    frame_cuts = load_frame_cuts(args.frame_cuts)
    sources = discover_sources(args.dataset_root, args.limit, args.contents)
    missing_cuts = sorted(source.id for source in sources if source.id not in frame_cuts)
    if missing_cuts:
        raise ValueError(f"Frame-cut CSV is missing selected sources: {missing_cuts[:8]}")
    if args.limit < 0:
        styles_by_content = {
            content: {source.style for source in sources if source.motion_code == content}
            for content in args.contents
        }
        cut_styles_by_content = {
            content: {
                source_id[: -(len(content) + 1)]
                for source_id in frame_cuts
                if source_id.endswith(f"_{content}")
            }
            for content in args.contents
        }
        expected_styles = cut_styles_by_content[args.contents[0]]
        mismatched = {
            content: sorted(cut_styles_by_content[content].symmetric_difference(styles))
            for content, styles in styles_by_content.items()
            if styles != cut_styles_by_content[content]
        }
        inconsistent_cuts = {
            content: sorted(expected_styles.symmetric_difference(styles))
            for content, styles in cut_styles_by_content.items()
            if styles != expected_styles
        }
        if not expected_styles or mismatched or inconsistent_cuts:
            raise ValueError(
                "Selected contents do not match the complete frame-cut style set: "
                f"source mismatches={mismatched}, cut mismatches={inconsistent_cuts}"
            )
    planned, errors = build_window_plan(
        sources,
        args.dataset_root,
        args.num_frames,
        args.fps,
        frame_cuts,
        args.min_frames,
        args.split_seed,
    )
    if errors:
        first = errors[0]
        raise RuntimeError(
            f"100STYLE window planning failed for {len(errors)} sources; "
            f"first error: {first['id']}: {first['error']}"
        )
    if not planned:
        raise RuntimeError("No 100STYLE windows meeting --min_frames were discovered")

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

    if errors:
        first = errors[0]
        raise RuntimeError(
            f"100STYLE conversion failed for {len(errors)} sources; "
            f"first error: {first['id']}: {first['error']}"
        )

    rng = np.random.default_rng(args.split_seed)
    for records in records_by_split.values():
        rng.shuffle(records)

    finalize_processed_dataset(
        args,
        records_by_split,
        errors,
        {split: len(sources_by_split[split]) for split in SPLITS},
        source_dataset="100STYLE_soma77",
        segmentation=SEGMENTATION_POLICY,
        preprocessing_version=PREPROCESSING_VERSION,
        num_source_motions=len(successful_sources),
        metadata_extra={
            "split_unit": "window",
            "source_pattern": SOURCE_PATTERN,
            "tail_policy": "retain_remainder_at_least_min_frames",
            "source_standard_tpose": True,
            "num_discovered_source_motions": len(sources),
            "downsampling": "fixed_step_from_trimmed_frame_zero_no_interpolation",
            **{key: value for key, value in protocol.items() if key != "version"},
            "preprocessing_protocol": protocol,
            "preprocessing_protocol_sha256": _protocol_sha256(protocol),
        },
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert 100STYLE SOMA77 BVHs to shuffled MotionJEPA NPY windows."
    )
    add_representation_argument(parser)
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=PROJECT_ROOT / "dataset/100style-soma77",
    )
    parser.add_argument(
        "--frame-cuts",
        type=Path,
        default=None,
        help="Frame-cut CSV (default: <dataset_root>/Frame_Cuts.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "dataset/100style-processed-nframes60-150",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, os.cpu_count() or 1),
        help="Worker processes. Each worker uses one PyTorch CPU thread.",
    )
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=150)
    parser.add_argument("--min_frames", "--min-frames", type=int, default=60)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--contents", nargs="+", default=list(DEFAULT_CONTENTS))
    parser.add_argument("--split_seed", "--split-seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        metavar="N",
        help="Process at most N source BVHs; negative values process every source.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    preprocess(parse_args())
