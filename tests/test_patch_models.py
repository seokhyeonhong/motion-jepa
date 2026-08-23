"""Tests for non-overlapping 1D temporal patch models."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from mask import PatchMaskCollator1D
from mask.utils import apply_index_masks, repeat_mask_blocks
from model import MotionPatchTransformer1D, MotionPatchTransformerPredictor1D
from model import mot_patch_tiny_1d
from experiment.linear_probe.features import load_frozen_encoder, pool_encoder_output
from experiment.linear_probe.train_classifier import _extract_token_features


class PatchModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.motion = torch.randn(2, 8, 6)
        self.fps = torch.full((2,), 60)

    def _encoder(self):
        return MotionPatchTransformer1D(
            6, 8, temporal_patch_size=3, embed_dim=24, depth=2, num_heads=3
        )

    def test_shape_tail_and_valid_patch_zeroing(self):
        encoder = self._encoder().eval()
        valid = torch.arange(8).unsqueeze(0) < torch.tensor([[8], [5]])
        with torch.no_grad():
            output = encoder(self.motion, self.fps, valid_frames=valid)
        self.assertEqual(output.shape, (2, 2, 24))
        self.assertEqual(torch.count_nonzero(output[1, 1]).item(), 0)
        self.assertEqual(encoder.token_layout.valid_token_lengths(torch.tensor([8, 5])).tolist(), [2, 1])

    def test_target_patch_values_cannot_leak_into_context(self):
        encoder = self._encoder().eval()
        changed = self.motion.clone()
        changed[:, 3:6] = torch.randn_like(changed[:, 3:6]) * 1000
        context_mask = [torch.zeros((2, 1), dtype=torch.long)]
        with torch.no_grad():
            original = encoder(self.motion, self.fps, context_mask)
            modified = encoder(changed, self.fps, context_mask)
        torch.testing.assert_close(original, modified)

    def test_multi_mask_forward_backward(self):
        batch = [(self.motion[index], 60, 8) for index in range(2)]
        _, masks_enc, masks_pred = PatchMaskCollator1D(
            8,
            3,
            enc_frame_mask_ratio=(0.5, 0.5),
            pred_frame_mask_ratio=(0.5, 0.5),
            nenc=2,
            npred=2,
            allow_overlap=True,
        )(batch)
        encoder = self._encoder()
        predictor = MotionPatchTransformerPredictor1D(
            8, 3, 24, 12, depth=2, num_heads=3
        )
        target = encoder(self.motion, self.fps)
        context = encoder(self.motion, self.fps, masks_enc)
        prediction = predictor(context, self.fps, masks_enc, masks_pred)
        expected = repeat_mask_blocks(
            apply_index_masks(target, masks_pred), len(self.motion), len(masks_enc)
        )
        self.assertEqual(prediction.shape, expected.shape)
        loss = F.smooth_l1_loss(prediction, expected.detach())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(encoder.patch_embed.weight.grad)

    def test_patch_checkpoint_supports_frozen_and_token_probes(self):
        encoder = mot_patch_tiny_1d(
            in_chans=6, num_frames=8, temporal_patch_size=3
        )
        config = {
            "data": {
                "num_frames": 8,
                "motion_dim": 6,
                "num_joints": 30,
                "fps": 60,
            },
            "patch": {"temporal_patch_size": 3},
            "meta": {
                "model_name": "mot_patch_tiny_1d",
                "predictor_name": "mot_predictor_patch_tiny_1d",
                "use_bfloat16": False,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "patch.pth.tar"
            torch.save(
                {
                    "format_version": 1,
                    "target_encoder": encoder.state_dict(),
                    "config": config,
                },
                checkpoint_path,
            )
            loaded, _, info = load_frozen_encoder(
                checkpoint_path, "target_encoder", torch.device("cpu")
            )
        self.assertEqual(info["token_num_frames"], 2)
        self.assertEqual(info["temporal_patch_size"], 3)
        pooled = pool_encoder_output(
            torch.tensor([[[1.0], [3.0]]]),
            torch.tensor([5]),
            loaded.token_layout,
        )
        torch.testing.assert_close(pooled, torch.tensor([[1.0]]))

        sample = (self.motion[0], torch.tensor(60), torch.tensor(5), torch.tensor(0), "x")
        payload = _extract_token_features(
            loaded,
            [sample],
            device=torch.device("cpu"),
            batch_size=1,
            num_workers=0,
        )
        self.assertEqual(payload["features"].shape[1], 2)
        torch.testing.assert_close(payload["lengths"], torch.tensor([1]))


if __name__ == "__main__":
    unittest.main()
