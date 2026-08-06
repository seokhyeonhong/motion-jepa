"""Tests for lazy NPY loading and standard distributed sampling."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from _npy_fixture import write_npy_dataset
from dataset import MotionDataset, make_motion_dataset


class NPYMotionDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.motions = [
            np.arange(3 * 2, dtype=np.float32).reshape(3, 2),
            np.full((4, 2), 5.0, dtype=np.float32),
            np.full((2, 2), -2.0, dtype=np.float32),
            np.full((4, 2), 9.0, dtype=np.float32),
            np.full((3, 2), 11.0, dtype=np.float32),
        ]
        write_npy_dataset(self.root, self.motions, num_frames=4)

    def tearDown(self):
        self.temporary.cleanup()

    def test_loading_normalization_and_tail_padding(self):
        stats = self.root / "stats"
        stats.mkdir()
        np.save(stats / "mean.npy", np.array([1.0, 2.0], dtype=np.float32))
        np.save(stats / "std.npy", np.array([2.0, 4.0], dtype=np.float32))
        dataset = MotionDataset(
            self.root,
            "train.txt",
            num_frames=4,
            fps=60,
            motion_dim=2,
            normalize=True,
        )
        actual, fps, length = dataset[0]
        np.testing.assert_allclose(
            actual[:3], (self.motions[0] - [1.0, 2.0]) / [2.0, 4.0]
        )
        np.testing.assert_array_equal(actual[3], np.zeros(2, dtype=np.float32))
        self.assertTrue(actual.flags.writeable)
        self.assertEqual((fps, length), (60, 3))

    def test_random_access_is_exact(self):
        dataset = MotionDataset(self.root, "train.txt", 4, 60, motion_dim=2)
        for index in (3, 0, 4, 1, 2):
            actual, _, length = dataset[index]
            np.testing.assert_array_equal(actual[:length], self.motions[index])
            np.testing.assert_array_equal(actual[length:], 0.0)

    def test_multiworker_loader_and_two_rank_sampler_have_equal_counts(self):
        dataset, loader, sampler = make_motion_dataset(
            self.root,
            ["train.txt"],
            batch_size=1,
            num_frames=4,
            fps=60,
            motion_dim=2,
            rank=0,
            world_size=1,
            num_workers=2,
            persistent_workers=False,
            pin_mem=False,
            drop_last=False,
        )
        seen = [(float(batch[0, 0, 0]), int(lengths[0])) for batch, _, lengths in loader]
        self.assertEqual(len(seen), len(self.motions))
        self.assertIsInstance(sampler, torch.utils.data.distributed.DistributedSampler)
        rank0 = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=2, rank=0, shuffle=True, seed=3
        )
        rank1 = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=2, rank=1, shuffle=True, seed=3
        )
        self.assertEqual(len(list(rank0)), len(list(rank1)))
        self.assertEqual(len(list(rank0)), 3)

    def test_stale_manifest_hash_is_rejected(self):
        with (self.root / "train.txt").open("a", encoding="utf-8") as file:
            file.write("extra,motions/train/sample-0.npy,60,3\n")
        with self.assertRaisesRegex(ValueError, "split hash is stale"):
            MotionDataset(self.root, "train.txt", 4, 60, motion_dim=2)


if __name__ == "__main__":
    unittest.main()
