"""Tests for 100STYLE window-level preprocessing."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dataset.motion_dataset import MotionDataset
from dataset import preprocess_100style as preprocessing


def _write_timing_header(path: Path, frames: int = 1800) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "HIERARCHY\nMOTION\n"
        f"Frames: {frames}\n"
        "Frame Time: 0.0166666667\n",
        encoding="utf-8",
    )


class WindowPlanningTest(unittest.TestCase):
    def test_style_motion_parsing_and_source_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_timing_header(root / "bvh/Zombie_FW_soma77.bvh")
            _write_timing_header(root / "bvh/Aeroplane_BR_soma77.bvh")
            _write_timing_header(root / "bvh/ignored.bvh")

            sources = preprocessing.discover_sources(root, limit=1)
            self.assertEqual(len(sources), 1)
            self.assertEqual(sources[0].id, "Aeroplane_BR")
            self.assertEqual(sources[0].style, "Aeroplane")
            self.assertEqual(sources[0].motion_code, "BR")
            self.assertEqual(sources[0].bvh_path, "bvh/Aeroplane_BR_soma77.bvh")

    def test_windows_are_non_overlapping(self):
        windows = preprocessing.enumerate_windows("style_FW", 365, 90)
        self.assertEqual(
            [(window.start_frame, window.end_frame) for window in windows],
            [(0, 90), (90, 180), (180, 270), (270, 360)],
        )
        for left, right in zip(windows, windows[1:]):
            self.assertLessEqual(left.end_frame, right.start_frame)

    def test_global_shuffle_is_reproducible_and_exact(self):
        windows = [
            preprocessing.WindowDescriptor("source", index, index * 90, (index + 1) * 90)
            for index in range(10)
        ]
        first = preprocessing.assign_window_splits(windows, seed=42)
        second = preprocessing.assign_window_splits(windows, seed=42)
        different = preprocessing.assign_window_splits(windows, seed=7)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        counts = {
            split: sum(window.split == split for window in first)
            for split in preprocessing.SPLITS
        }
        self.assertEqual(counts, {"train": 8, "val": 1, "test": 1})
        self.assertEqual(
            {(window.source_id, window.segment_index) for window in first},
            {(window.source_id, window.segment_index) for window in windows},
        )

        uneven = [
            preprocessing.WindowDescriptor("source", index, index * 90, (index + 1) * 90)
            for index in range(326)
        ]
        uneven_assignment = preprocessing.assign_window_splits(uneven, 42)
        uneven_counts = {
            split: sum(window.split == split for window in uneven_assignment)
            for split in preprocessing.SPLITS
        }
        self.assertEqual(uneven_counts, {"train": 261, "val": 33, "test": 32})


class Preprocess100StyleOutputTest(unittest.TestCase):
    def test_output_is_motion_dataset_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "100STYLE_soma77"
            output = root / "processed"
            _write_timing_header(dataset_root / "bvh/Aeroplane_BR_soma77.bvh")
            args = argparse.Namespace(
                dataset_root=dataset_root,
                output=output,
                workers=1,
                chunksize=1,
                num_frames=90,
                fps=30,
                split_seed=42,
                limit=-1,
                overwrite=False,
            )

            def fake_results(items, _args):
                for item in items:
                    records = []
                    for window in item.windows:
                        value = float(window.segment_index + 1)
                        records.append(
                            {
                                "id": f"{item.source.id}_{window.segment_index:04d}",
                                "source_id": item.source.id,
                                "segment_index": window.segment_index,
                                "start_frame": window.start_frame,
                                "end_frame": window.end_frame,
                                "split": window.split,
                                "source_path": item.source.bvh_path,
                                "source_fps": 60,
                                "fps": 30,
                                "length": 90,
                                "motion_dim": 366,
                                "metadata": {
                                    "style": item.source.style,
                                    "motion_code": item.source.motion_code,
                                    "source_id": item.source.id,
                                    "start_frame": window.start_frame,
                                    "end_frame": window.end_frame,
                                },
                                "motion": np.full((90, 366), value, dtype=np.float32),
                            }
                        )
                    yield {"ok": True, "id": item.source.id, "records": records}

            with mock.patch.object(
                preprocessing, "_ordered_results", side_effect=fake_results
            ):
                preprocessing.preprocess(args)

            metadata = json.loads((output / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["source_dataset"], "100STYLE_soma77")
            self.assertEqual(metadata["split_unit"], "window")
            self.assertEqual(metadata["overlap"], 0.0)
            self.assertEqual(metadata["stride_frames"], 90)
            self.assertTrue(metadata["source_standard_tpose"])
            self.assertEqual(metadata["split_counts"], {"train": 8, "val": 1, "test": 1})

            index = json.loads((output / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(len(index), 10)
            self.assertTrue(all(record["metadata"]["style"] == "Aeroplane" for record in index))
            self.assertTrue(all(record["metadata"]["motion_code"] == "BR" for record in index))

            train = MotionDataset(output, "train.txt", 90, 30, motion_dim=366)
            self.assertEqual(len(train), 8)
            motion, fps, length = train[0]
            self.assertEqual(motion.shape, (90, 366))
            self.assertEqual((fps, length), (30, 90))


if __name__ == "__main__":
    unittest.main()
