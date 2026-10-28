"""Layer concatenation, patch alignment, and training/resume contracts."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from experiment import probing
from experiment.segmentation import probe, train
from experiment.segmentation.model import MotionSegmentationPatchTransformerProbe
from model.motion_transformer_1d import MotionTransformer1D


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    def factory(in_chans, num_frames, patchify):
        return MotionTransformer1D(in_chans=in_chans, num_frames=num_frames,
                                   embed_dim=12, depth=4, num_heads=2, patchify=patchify)
    monkeypatch.setattr(probing.motion_transformer_1d, "mot_concat_fixture", factory, raising=False)
    model = factory(6, 6, True)
    path = tmp_path / "encoder.pth"
    torch.save({"config": {"data": {"motion_dim": 6, "num_frames": 6, "normalize": False},
                          "meta": {"model_name": "mot_concat_fixture", "patchify": True}},
                "target_encoder": model.state_dict(), "global_step": 200000}, path)
    return path


def test_single_pass_order_matches_independent_layer_outputs(checkpoint):
    ex = probe.FrozenPatchLayersExtractor(checkpoint, [4, 1, 3], device="cpu")
    x = torch.randn(2, 6, 6)
    lengths = torch.tensor([4, 6])
    fps = torch.tensor([30., 30.])
    calls = [0] * 4
    def count(index):
        def hook(*args):
            calls[index] += 1
        return hook
    hooks = [b.register_forward_hook(count(i)) for i, b in enumerate(ex.encoder.blocks)]
    result = ex(x, fps, lengths)
    assert calls == [1, 1, 1, 1]
    for hook in hooks:
        hook.remove()
    valid = torch.arange(6)[None] < lengths[:, None]
    normalized = x.masked_fill(~valid[..., None], 0)
    encoder_valid = valid.reshape(2, 2, 3).any(-1).repeat_interleave(3, 1)
    with torch.inference_mode():
        expected = torch.cat([ex.encoder(normalized, fps, valid_frames=encoder_valid,
                                         feature_layer=i) for i in [4, 1, 3]], -1)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert result.shape == (2, 2, 36) and not result.requires_grad
    assert all(not p.requires_grad for p in ex.encoder.parameters())
    assert ex.metadata["feature_layers"] == [4, 1, 3]
    assert all(not b._forward_hooks for b in ex.encoder.blocks)
    x[0, 4:] = float("nan")
    torch.testing.assert_close(ex(x, fps, lengths), result)
    ex.release()
    with pytest.raises(RuntimeError, match="released"):
        ex(x, fps, lengths)


def test_hook_cleanup_on_failure(checkpoint, monkeypatch):
    ex = probe.FrozenPatchLayersExtractor(checkpoint, [1, 4])
    def fail(*args, **kwargs):
        raise RuntimeError("fixture failure")
    monkeypatch.setattr(ex.encoder.blocks[2], "forward", fail)
    with pytest.raises(RuntimeError, match="fixture failure"):
        ex(torch.zeros(1, 6, 6), torch.tensor([30.]), torch.tensor([6]))
    assert all(not b._forward_hooks for b in ex.encoder.blocks)


@pytest.mark.parametrize("layers", [[], [1, 1], [0], [-1], [True], [1.5]])
def test_invalid_layers(layers):
    with pytest.raises(ValueError):
        probe.selected_feature_layers(SimpleNamespace(feature_layers=layers))


def test_out_of_range_and_nonpatch_checkpoint(checkpoint):
    with pytest.raises(ValueError, match="block index"):
        probe.FrozenPatchLayersExtractor(checkpoint, [1, 5])
    saved = torch.load(checkpoint, weights_only=False)
    saved["config"]["meta"]["patchify"] = False
    factory = probing.motion_transformer_1d.mot_concat_fixture
    saved["target_encoder"] = factory(6, 6, False).state_dict()
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="patch JEPA"):
        probe.FrozenPatchLayersExtractor(checkpoint, [1])


@pytest.mark.parametrize("layers,parameters", [(1, 2061672), (2, 2209128), (4, 2504040)])
def test_model_width_counts_masking_and_tail_gradients(layers, parameters):
    width = 384 * layers
    model = MotionSegmentationPatchTransformerProbe(width, 120, 6, torch.zeros(width), torch.ones(width))
    assert model.parameter_counts()["total"] == parameters
    assert len(model.encoder.blocks) == 1 and model.encoder.embed_dim == 384
    x = torch.randn(2, 2, width, requires_grad=True)
    valid = torch.tensor([[True, True, True, True, False, False], [True] * 6])
    output = model(x, valid, torch.tensor([30., 30.]))
    assert output.shape == (2, 6, 120) and not output[~valid].any()
    output[0, 3].square().mean().backward()
    assert x.grad[0, 1].abs().sum() > 0
    assert torch.isfinite(x.grad).all()
    valid[0, 3] = False
    a = x.detach().clone()
    b = a.clone()
    b[0, 1] = float("nan")
    torch.testing.assert_close(model(a, valid, torch.tensor([30., 30.])),
                               model(b, valid, torch.tensor([30., 30.])))
    clone = MotionSegmentationPatchTransformerProbe(width, 120, 6, torch.zeros(width), torch.ones(width))
    clone.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(model(a, valid, torch.tensor([30., 30.])),
                               clone(a, valid, torch.tensor([30., 30.])))


class PatchFrames(Dataset):
    num_frames = 6
    motion_dim = 6
    num_classes = 2
    class_indices_60 = (0,)

    def __init__(self, split):
        self.split = split
        self.count = {"train": 3, "val": 2, "test": 0}[split]

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        length = [4, 6, 3][index]
        motion = torch.arange(36).reshape(6, 6).float() / 10 + index
        if self.split == "val":
            motion += 10
        labels = torch.zeros(6, 2)
        labels[:length, index % 2] = 1
        return motion, 30., length, labels, labels.any(-1), f"{self.split}-{index}"


def args_for(checkpoint):
    return SimpleNamespace(feature_source="jepa", checkpoint=checkpoint, feature_layer=None,
                           feature_layers=[1, 3], encoder_key="target_encoder",
                           precompute_batch_size=2, probe_heads=6, num_workers=0,
                           probe_architecture="patch-transformer")


def test_cache_separate_channel_stats_and_no_validation_leakage(checkpoint):
    datasets = {s: PatchFrames(s) for s in ("train", "val", "test")}
    cached, mean, std, metadata = probe.prepare_probe_datasets(
        datasets, args_for(checkpoint), torch.device("cpu"))
    valid = torch.arange(2)[None] < (cached["train"].lengths[:, None] + 2) // 3
    features = cached["train"].features[valid]
    expected_std = features.std(0, correction=0)
    expected_std = torch.where(expected_std < 1e-6, torch.ones_like(expected_std), expected_std)
    torch.testing.assert_close(mean, features.mean(0))
    torch.testing.assert_close(std, expected_std)
    assert mean.shape == (24,) and metadata["training_tokens"] == 5
    assert cached["train"].features.shape == (3, 2, 24)
    assert cached["train"].labels.shape == (3, 6, 2)
    assert metadata["layers"] == [1, 3]
    assert cached["val"].sample_ids == ["val-0", "val-1"]


@pytest.mark.parametrize("architecture", ["patch-transformer", "mlp"])
def test_cli_and_resume_layer_order_contract(tmp_path, monkeypatch, checkpoint, architecture):
    root = tmp_path / "data"
    root.mkdir()
    (root / "stats").mkdir()
    (root / "meta.json").write_text(json.dumps(dict(motion_dim=6, num_frames=6, fps=30)))
    for name in ("index.json", "train.txt", "val.txt", "test.txt"):
        (root / name).write_text("fixture")
    np.save(root / "stats/mean.npy", np.zeros(6))
    np.save(root / "stats/std.npy", np.ones(6))
    monkeypatch.setattr(train, "BabelSegmentationDataset", lambda root, split, **kw: PatchFrames(split))
    args = train.build_parser().parse_args([
        "--mode", "probe", "--probe-architecture", architecture,
        "--feature-layers", "1", "3", "--checkpoint", str(checkpoint),
        "--dataset-root", str(root), "--output-root", str(tmp_path / "run"),
        "--device", "cpu", "--iterations", "2", "--warmup-iterations", "0",
        "--eval-interval", "1", "--checkpoint-interval", "1",
        "--batch-size", "2", "--num-workers", "0",
    ])
    summary = train.run(args)
    assert summary["feature_layers"] == [1, 3]
    assert summary["iterations_completed"] == 2
    saved = torch.load(tmp_path / "run/latest.pth.tar", weights_only=False)
    assert saved["signature"]["features"]["feature_layers"] == [1, 3]
    assert saved["model"]["feature_mean"].shape == (24,)
    args.resume = True
    assert train.run(args)["best_step"] == summary["best_step"]
    args.feature_layers = [3, 1]
    with pytest.raises(ValueError, match="Resume dataset"):
        train.run(args)
    with pytest.raises(SystemExit):
        train.build_parser().parse_args(["--feature-layer", "3", "--feature-layers", "3", "6"])


def test_single_layer_cli_legacy_alias(checkpoint):
    args = train.build_parser().parse_args(["--mode", "probe", "--probe-architecture",
         "patch-transformer", "--feature-layer", "3", "--checkpoint", str(checkpoint)])
    train._validate_motion_depth(args)
    assert probe.selected_feature_layers(args) == [3]
