"""Focused architecture and cache regressions; no model downloads required."""

import dataclasses
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from .dataset import DEFAULT_TEXT_MODEL, RaggedTokenBank
from .features import _prepare_text, _text_signature
from .model import (
    AlignmentConfig, TextMotionAlignment, MODEL_SPECS,
)
from .train import build_parser, _validate_args


class JEPAEncoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_all_size_presets_and_checkpoint_config(self):
        for size, spec in MODEL_SPECS.items():
            config = AlignmentConfig(text_dim=4096, motion_dim=12, num_frames=8, fps=30, model_size=size)
            self.assertEqual(dataclasses.asdict(config) | spec, dataclasses.asdict(config))
            self.assertEqual(AlignmentConfig(**dataclasses.asdict(config)), config)
            with patch("model.motion_transformer_1d.torch.linspace", return_value=torch.zeros(spec["depth"])), torch.device("meta"):
                model = TextMotionAlignment(config)
            for encoder in (model.text_encoder, model.motion_encoder):
                from model.motion_transformer_1d import MotionTransformer1D
                self.assertIsInstance(encoder.backbone, MotionTransformer1D)
                self.assertEqual(len(encoder.backbone.blocks), spec["depth"])
                self.assertEqual(encoder.backbone.input_proj.out_features, spec["embed_dim"])
                self.assertEqual(encoder.backbone.num_heads, spec["num_heads"])
                self.assertEqual(encoder.backbone.rotary.theta, 100.0)
                self.assertEqual(encoder.backbone.rotary.time_scale, 1.0)
                self.assertEqual(encoder.backbone.norm.eps, 1e-6)
                self.assertEqual(encoder.backbone.blocks[0].attn.attn_drop, 0.0)

    def test_padding_gradients_and_state_roundtrip(self):
        config = AlignmentConfig(text_dim=4096, motion_dim=12, num_frames=8, fps=30, text_num_tokens=4, model_size="tiny")
        model = TextMotionAlignment(config).eval()
        motion = torch.randn(2, 5, 12)
        text = torch.randn(2, 1, 4096)
        motion_mask = torch.ones(2, 5, dtype=torch.bool)
        text_mask = torch.ones(2, 1, dtype=torch.bool)
        a, b = model(motion, motion_mask, text, text_mask)
        self.assertEqual(a.shape, (2, 256))
        self.assertEqual(b.shape, (2, 256))
        torch.testing.assert_close(a.norm(dim=-1), torch.ones(2))
        torch.testing.assert_close(b.norm(dim=-1), torch.ones(2))
        for values, mask, encode, expected in (
            (motion, motion_mask, model.encode_motion, a),
            (text, text_mask, model.encode_text, b),
        ):
            padded = torch.cat((values, torch.full((2, 3, values.shape[-1]), float("nan"))), 1)
            padded_mask = torch.cat((mask, torch.zeros(2, 3, dtype=torch.bool)), 1)
            torch.testing.assert_close(encode(padded, padded_mask), expected, atol=2e-6, rtol=2e-5)
            with self.assertRaises(ValueError):
                encode(values, torch.zeros_like(mask))
        loss = torch.nn.functional.cross_entropy(a @ b.T / .1, torch.arange(2))
        loss.backward()
        for encoder in (model.text_encoder, model.motion_encoder):
            self.assertGreater(encoder.backbone.input_proj.weight.grad.abs().sum().item(), 0)
            self.assertGreater(encoder.backbone.blocks[0].attn.qkv.weight.grad.abs().sum().item(), 0)
        restored = TextMotionAlignment(AlignmentConfig(**dataclasses.asdict(config))).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(restored.encode_text(text, text_mask), b)

    def test_matches_shared_jepa_backbone_and_modality_positions(self):
        from model.motion_transformer_1d import MotionTransformer1D

        config = AlignmentConfig(
            text_dim=16, motion_dim=12, num_frames=8, fps=20, text_num_tokens=4,
            model_size="tiny",
        )
        model = TextMotionAlignment(config).eval()
        for encoder, width, length, rate in (
            (model.text_encoder, 16, 4, 1.0), (model.motion_encoder, 12, 8, 20.0),
        ):
            reference = MotionTransformer1D(
                in_chans=width, num_frames=length, **MODEL_SPECS["tiny"],
            ).eval()
            reference.load_state_dict(encoder.backbone.state_dict(), strict=True)
            values = torch.randn(2, length, width)
            mask = torch.arange(length)[None] < torch.tensor([length, length - 1])[:, None]
            positions = []
            handle = encoder.backbone.rotary.register_forward_pre_hook(
                lambda module, args: positions.append(args[0].detach().clone()),
            )
            with torch.no_grad():
                actual = encoder(values, mask)
                encoded = reference(
                    values.masked_fill(~mask[..., None], 0),
                    torch.full((2,), rate), valid_frames=mask,
                )
                expected = torch.nn.functional.normalize(
                    encoder.output_projection(encoded.sum(1) / mask.sum(1, keepdim=True)), dim=-1,
                )
            handle.remove()
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(positions[0], torch.arange(length).float().expand(2, -1) / rate)
            with self.assertRaises(ValueError):
                encoder(torch.randn(2, length + 1, width), torch.ones(2, length + 1, dtype=torch.bool))

    def test_configuration_has_no_version_or_legacy_fields(self):
        config = AlignmentConfig(text_dim=4096, motion_dim=12, num_frames=8, fps=30)
        self.assertNotIn("model_version", dataclasses.asdict(config))
        for field in ("model_version", "text_pooling", "motion_pooling", "dropout"):
            with self.assertRaises(TypeError):
                AlignmentConfig(**{**dataclasses.asdict(config), field: "removed"})

    def test_cli_defaults_and_removed_options(self):
        parser = build_parser()
        args = parser.parse_args([])
        _validate_args(args)
        self.assertEqual(args.model_size, "base")
        self.assertFalse(hasattr(args, "model_version"))
        self.assertEqual(args.text_model, DEFAULT_TEXT_MODEL)
        for option in (
            "--model-version", "--depth", "--text-depth", "--motion-depth",
            "--embed-dim", "--text-pooling", "--motion-pooling", "--num-heads",
            "--ff-dim", "--dropout",
        ):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                parser.parse_args([option, "6"])


    def test_training_resume_and_evaluation_use_unversioned_config(self):
        from . import train, evaluate

        class Pairs(torch.utils.data.Dataset):
            caption_candidates = [["a"], ["b"]]
            text_bank = {"a": torch.randn(1, 4096), "b": torch.randn(1, 4096)}
            motions = torch.randn(2, 4, 12)

            def __len__(self):
                return 2

            def __getitem__(self, index):
                caption = self.caption_candidates[index][0]
                return {
                    "motion_tokens": self.motions[index],
                    "text_tokens": self.text_bank[caption],
                    "caption_id": caption, "caption_candidate_ids": [caption],
                    "sample_id": str(index), "source_id": str(index),
                    "start_frame": 0, "end_frame": 4,
                }

        data = {split: Pairs() for split in ("train", "val", "test")}
        provenance = {"text_dim": 4096, "motion_dim": 12, "provenance": {"num_frames": 4, "fps": 30}}
        args = build_parser().parse_args([
            "--model-size", "tiny", "--device", "cpu", "--epochs", "1",
            "--warmup-epochs", "0", "--batch-size", "2", "--eval-batch-size", "2",
            "--num-workers", "0", "--no-tensorboard",
        ])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            with patch.object(train, "PROJECT_ROOT", Path(directory)), patch.object(
                train, "load_prepared_datasets", return_value=(data, provenance),
            ):
                summary = train.run(args)
                expected = Path(directory) / "output/tmr/tiny/seed-42/best.pth.tar"
                self.assertEqual(Path(summary["best_checkpoint"]), expected)
                checkpoint = train.load_alignment_checkpoint(expected)
                self.assertNotIn("model_version", checkpoint["config"]["model"])
                args.resume = True
                self.assertEqual(train.run(args)["best_epoch"], summary["best_epoch"])
            with patch.object(evaluate, "load_prepared_datasets", return_value=(data, provenance)):
                metrics = evaluate.run(evaluate.build_parser().parse_args([
                    "--checkpoint", str(expected), "--device", "cpu", "--num-workers", "0",
                ]))
                self.assertEqual(metrics, summary["test"])


class LLM2VecTests(unittest.TestCase):
    def test_future_tokens_affect_first_token_and_padding_is_ignored(self):
        from transformers import LlamaConfig, LlamaModel
        from .llm2vec import LLM2VecBackbone
        torch.manual_seed(3)
        config = LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        )
        config._attn_implementation = "sdpa"
        backbone = LLM2VecBackbone(LlamaModel(config).eval(), None)
        ids = torch.tensor([[1, 2, 3]])
        mask = torch.ones_like(ids)
        with torch.no_grad():
            hidden = backbone.hidden_states(ids, mask)
            changed = backbone.hidden_states(torch.tensor([[1, 2, 4]]), mask)
            self.assertFalse(torch.allclose(hidden[:, 0], changed[:, 0], atol=1e-6))
            padded = backbone.hidden_states(torch.tensor([[1, 2, 3, 7, 8]]), torch.tensor([[1, 1, 1, 0, 0]]))
            torch.testing.assert_close(hidden, padded[:, :3])

    def test_sentence_cache_and_adapter_provenance(self):
        paired = {"provenance": {}, "catalog_sha256": "test", "catalog": {"a": "walk", "b": "run"}}
        signature = _text_signature(paired, DEFAULT_TEXT_MODEL, None, 256)
        self.assertNotEqual(signature, _text_signature(paired, DEFAULT_TEXT_MODEL, None, 256, adapter_revision="new"))
        self.assertNotEqual(signature, _text_signature(paired, DEFAULT_TEXT_MODEL, None, 256, adapter="different-adapter"))

        class FakeBackbone:
            def token_length(self, text, max_length):
                return 5
            def encode(self, texts, max_length, device):
                return [torch.full((1, 4096), float(len(text))) for text in texts]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "text"
            with patch("experiment.tmr.features.load_text_backbone", return_value=(
                FakeBackbone(), None, {"model_class": "LLM2VecBackbone"},
            )):
                bank = _prepare_text(
                    path, paired, model_name=DEFAULT_TEXT_MODEL, revision=None,
                    max_length=256, device=torch.device("cpu"), batch_size=2, recompute=False,
                )
            self.assertEqual(bank["a"].shape, (1, 4096))
            with self.assertRaises(ValueError):
                RaggedTokenBank(path, _text_signature(paired, DEFAULT_TEXT_MODEL, None, 128))


if __name__ == "__main__":
    unittest.main()
