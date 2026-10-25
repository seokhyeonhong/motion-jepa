"""Temporal patch pooling and checkpoint compatibility for raw TMR."""
from dataclasses import asdict
import pytest
import torch
from torch import nn
from . import model as module
from .train import build_parser, _validate_args

@pytest.fixture
def config(monkeypatch):
    monkeypatch.setitem(module.MODEL_SPECS, "base", dict(embed_dim=12, depth=2, num_heads=3))
    return module.AlignmentConfig(text_dim=8, motion_dim=5, num_frames=10, fps=30, patchify=True)

def test_patch_pooling_padding_and_gradients(config):
    torch.manual_seed(42)
    model = module.TextMotionAlignment(config).eval()
    encoder = model.motion_encoder
    assert isinstance(encoder.backbone.input_proj, nn.Conv1d)
    assert isinstance(model.text_encoder.backbone.input_proj, nn.Linear)
    x = torch.randn(2, 10, 5)
    mask = torch.arange(10)[None] < torch.tensor([7, 10])[:, None]
    actual = model.encode_motion(x, mask)
    active = encoder.backbone.token_layout.valid_token_mask(mask)
    assert active.sum(1).tolist() == [2, 3]
    encoded = encoder.backbone(x.masked_fill(~mask[..., None], 0), torch.full((2,), 30.), valid_frames=mask)
    expected = torch.nn.functional.normalize(encoder.output_projection(encoded.sum(1) / active.sum(1, keepdim=True)).float(), dim=-1)
    torch.testing.assert_close(actual, expected)
    changed = x.clone()
    changed[0, 6:] = float("nan")
    # The valid trailing frame must remain finite, but its value cannot affect complete patches.
    changed[0, 6] = 999
    changed[1, 9] = 999
    torch.testing.assert_close(model.encode_motion(changed, mask), actual)
    actual.square().sum().backward()
    assert encoder.backbone.input_proj.weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with pytest.raises(ValueError, match="complete motion patch"):
        model.encode_motion(x[:, :2], torch.ones(2, 2, dtype=torch.bool))

def test_checkpoint_roundtrip_and_legacy(config, tmp_path):
    model = module.TextMotionAlignment(config).eval()
    path = tmp_path / "checkpoint.pt"
    torch.save(dict(config=asdict(config), model=model.state_dict()), path)
    saved = torch.load(path, weights_only=True)
    restored = module.TextMotionAlignment(module.AlignmentConfig(**saved["config"])).eval()
    restored.load_state_dict(saved["model"], strict=True)
    x = torch.randn(2, 7, 5); mask = torch.ones(2, 7, dtype=torch.bool)
    torch.testing.assert_close(model.encode_motion(x, mask), restored.encode_motion(x, mask))
    old = asdict(config); old.pop("patchify")
    legacy = module.TextMotionAlignment(module.AlignmentConfig(**old))
    assert isinstance(legacy.motion_encoder.backbone.input_proj, nn.Linear)
    with pytest.raises(RuntimeError):
        legacy.load_state_dict(saved["model"], strict=True)

def test_cli():
    parser = build_parser()
    assert not parser.parse_args([]).patchify
    assert parser.parse_args(["--patchify"]).patchify
    assert not parser.parse_args(["--patchify", "--no-patchify"]).patchify
    with pytest.raises(ValueError, match="supervised"):
        _validate_args(parser.parse_args(["--mode", "probe", "--patchify"]))


def test_patchify_data_configuration():
    from .train import data_configuration
    args = build_parser().parse_args(["--patchify", "--representation", "LocalMotionJEPARep"])
    data = data_configuration(args)
    assert "local-from366" in data["dataset_root"]
    assert "patchify" not in data
