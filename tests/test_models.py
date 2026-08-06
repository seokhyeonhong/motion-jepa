"""Unit tests for the frame and skeletal-temporal JEPA variants."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from mask import MaskCollator1D, MaskCollator2D
from mask.utils import apply_index_masks, gather_grid_masks, repeat_mask_blocks
from model import (
    MotionFeatureTokenizer2D,
    MotionTransformer1D,
    MotionTransformer2D,
    MotionTransformerPredictor1D,
    MotionTransformerPredictor2D,
)
from skeleton import SOMASkeleton30
from train import update_ema


class SemanticTokenizerTest(unittest.TestCase):
    def test_every_feature_is_routed_once_to_its_semantic_joint(self):
        tokenizer = MotionFeatureTokenizer2D(embed_dim=12)
        motion = torch.arange(366, dtype=torch.float32).reshape(1, 1, 366)
        routed = tokenizer.split_features(motion)
        flattened = torch.cat(routed, dim=-1).flatten()
        torch.testing.assert_close(flattened.sort().values, torch.arange(366.0))
        torch.testing.assert_close(
            routed[0].flatten(),
            torch.tensor([0, 1, 2, 3, 4, *range(92, 98), *range(272, 275)]).float(),
        )
        torch.testing.assert_close(
            routed[1].flatten(),
            torch.tensor([*range(5, 8), *range(98, 104), *range(275, 278)]).float(),
        )
        skeleton = SOMASkeleton30(load=False)
        for contact, joint in enumerate(skeleton.foot_joint_idx):
            self.assertEqual(float(routed[joint][..., -1]), 362.0 + contact)


class ModelShapeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.batch_size = 2
        self.frames = 8
        self.motion = torch.randn(self.batch_size, self.frames, 366)
        self.fps = torch.full((self.batch_size,), 60)
        self.batch = [(self.motion[index], 60) for index in range(self.batch_size)]

    def test_1d_multi_mask_forward_backward(self):
        _, masks_enc, masks_pred = MaskCollator1D(
            self.frames,
            enc_frame_mask_ratio=(0.5, 0.5),
            pred_frame_mask_ratio=(0.25, 0.25),
            nenc=2,
            npred=2,
        )(self.batch)
        encoder = MotionTransformer1D(
            366, self.frames, embed_dim=24, depth=2, num_heads=3
        )
        predictor = MotionTransformerPredictor1D(
            self.frames, 24, 12, depth=2, num_heads=3
        )
        target = encoder(self.motion, self.fps)
        context = encoder(self.motion, self.fps, masks_enc)
        prediction = predictor(context, self.fps, masks_enc, masks_pred)
        expected = repeat_mask_blocks(
            apply_index_masks(target, masks_pred), self.batch_size, len(masks_enc)
        )
        self.assertEqual(prediction.shape, expected.shape)
        loss = F.smooth_l1_loss(prediction, expected.detach())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(encoder.input_proj.weight.grad)

    def test_2d_multi_mask_forward_backward_and_overlap(self):
        _, masks_enc, masks_pred = MaskCollator2D(
            self.frames,
            30,
            enc_frame_mask_ratio=(1.0, 1.0),
            enc_joint_mask_ratio=(1.0, 1.0),
            pred_frame_mask_ratio=(0.25, 0.25),
            pred_joint_mask_ratio=(0.25, 0.25),
            nenc=2,
            npred=2,
            allow_overlap=True,
        )(self.batch)
        encoder = MotionTransformer2D(
            366, self.frames, 30, embed_dim=24, depth=2, num_heads=3
        )
        predictor = MotionTransformerPredictor2D(
            self.frames, 30, 24, 12, depth=2, num_heads=3
        )
        target = encoder(self.motion, self.fps)
        context = encoder(self.motion, self.fps, masks_enc)
        prediction = predictor(context, self.fps, masks_enc, masks_pred)
        expected = repeat_mask_blocks(
            gather_grid_masks(target, masks_pred), self.batch_size, len(masks_enc)
        )
        self.assertEqual(context.shape, (4, self.frames, 30, 24))
        self.assertEqual(prediction.shape, expected.shape)
        loss = F.smooth_l1_loss(prediction, expected.detach())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(encoder.tokenizer.projections[0].weight.grad)

    def test_ema_updates_target_toward_online_encoder(self):
        online = torch.nn.Linear(2, 2, bias=False)
        target = torch.nn.Linear(2, 2, bias=False)
        online.weight.data.fill_(1.0)
        target.weight.data.zero_()
        update_ema(online, target, 0.9)
        torch.testing.assert_close(target.weight, torch.full_like(target.weight, 0.1))

    def test_mixed_length_masks_never_select_padding(self):
        batch = [(self.motion[0], 60, 8), (self.motion[1], 60, 5)]
        for collator in (
            MaskCollator1D(
                self.frames,
                enc_frame_mask_ratio=(0.5, 0.5),
                pred_frame_mask_ratio=(0.25, 0.25),
            ),
            MaskCollator2D(
                self.frames,
                enc_frame_mask_ratio=(0.5, 0.5),
                pred_frame_mask_ratio=(0.25, 0.25),
            ),
        ):
            _, contexts, targets = collator(batch)
            if isinstance(collator, MaskCollator1D):
                for mask in contexts + targets:
                    self.assertTrue((mask[1] < 5).all())
            else:
                for mask in contexts + targets:
                    self.assertFalse(mask[1, 5:].any())

    def test_padded_values_do_not_change_valid_encoder_outputs(self):
        valid_frames = torch.arange(self.frames).unsqueeze(0) < torch.tensor([[5], [8]])
        changed = self.motion.clone()
        changed[0, 5:] = torch.randn_like(changed[0, 5:]) * 1000
        for encoder in (
            MotionTransformer1D(366, self.frames, embed_dim=24, depth=2, num_heads=3),
            MotionTransformer2D(366, self.frames, 30, embed_dim=24, depth=2, num_heads=3),
        ):
            encoder.eval()
            with torch.no_grad():
                original = encoder(self.motion, self.fps, valid_frames=valid_frames)
                modified = encoder(changed, self.fps, valid_frames=valid_frames)
            torch.testing.assert_close(original[0, :5], modified[0, :5])
            self.assertEqual(torch.count_nonzero(original[0, 5:]).item(), 0)


if __name__ == "__main__":
    unittest.main()
