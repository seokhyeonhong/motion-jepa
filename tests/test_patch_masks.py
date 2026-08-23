"""Tests for temporal-patch mask collation."""

from __future__ import annotations

import unittest

import torch

from mask import PatchMaskCollator1D


class PatchMaskCollatorTest(unittest.TestCase):
    def test_mixed_lengths_map_to_complete_patches(self):
        collator = PatchMaskCollator1D(
            8,
            3,
            enc_frame_mask_ratio=(1.0, 1.0),
            pred_frame_mask_ratio=(0.5, 0.5),
            allow_overlap=True,
        )
        batch = [(torch.zeros(8, 6), 60, 8), (torch.zeros(8, 6), 60, 5)]
        collated, contexts, targets = collator(batch)
        torch.testing.assert_close(collated[2], torch.tensor([8, 5]))
        for mask in contexts + targets:
            self.assertTrue((mask[0] < 2).all())
            self.assertTrue((mask[1] < 1).all())

    def test_state_restores_next_patch_mask(self):
        kwargs = dict(
            raw_num_frames=12,
            temporal_patch_size=3,
            enc_frame_mask_ratio=(0.5, 0.5),
            pred_frame_mask_ratio=(0.25, 0.25),
            nenc=2,
            npred=2,
        )
        batch = [(torch.zeros(12, 6), 60, 12) for _ in range(2)]
        first = PatchMaskCollator1D(**kwargs)
        first(batch)
        state = first.state_dict()
        expected = first(batch)[1:]
        restored = PatchMaskCollator1D(**kwargs)
        restored.load_state_dict(state)
        actual = restored(batch)[1:]
        for expected_group, actual_group in zip(expected, actual):
            for expected_mask, actual_mask in zip(expected_group, actual_group):
                torch.testing.assert_close(actual_mask, expected_mask)

    def test_rejects_samples_without_a_complete_patch(self):
        collator = PatchMaskCollator1D(8, 3)
        with self.assertRaisesRegex(ValueError, "complete temporal patch"):
            collator([(torch.zeros(8, 6), 60, 2)])


if __name__ == "__main__":
    unittest.main()
