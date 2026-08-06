"""Dataset discovery and decoding tests shared by both viewers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from _npy_fixture import write_npy_dataset
from dataset.visualize import _display_path
from visualization import discover_entries, load_motion


class VisualizationDatasetTest(unittest.TestCase):
    def test_npy_discovery_label_caption_and_decode(self):
        features = np.load(
            Path(__file__).parent / "assets/motion_jepa_golden.npz",
            allow_pickle=False,
        )["features"].astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            write_npy_dataset(root, [features], num_frames=len(features))
            (root / "index.json").write_text(
                json.dumps(
                    [
                        {
                            "id": "sample-0",
                            "split": "train",
                            "captions": ["walk forward"],
                            "motion_path": "motions/train/sample-0.npy",
                            "length": len(features),
                        }
                    ]
                ),
                encoding="utf-8",
            )
            entries = discover_entries(root, "train", limit=0)
            self.assertEqual(len(entries), 1)
            entry = entries[0]
            self.assertEqual(entry.caption, "walk forward")
            self.assertEqual(
                _display_path(root, entry),
                "motions/train/sample-0.npy",
            )
            decoded, fps = load_motion(entry, fps=60)
            self.assertEqual(fps, 60)
            self.assertEqual(decoded["posed_joints"].shape[:2], (8, 77))

            (root / "train.txt").unlink()
            from_index = discover_entries(root, "train", limit=0)
            self.assertEqual(from_index[0].path, entry.path)
            self.assertEqual(from_index[0].actual_length, len(features))



if __name__ == "__main__":
    unittest.main()
