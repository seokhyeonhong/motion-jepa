import numpy as np
import pytest
import torch

from experiment.probing import AttentiveProbe, FeatureMoments, FrozenFrameExtractor
from model.motion_transformer_1d import MotionTransformer1D, mot_tiny_1d


def test_feature_moments_ignore_padding_and_match_full_population():
    x = torch.tensor([[[1., 5.], [3., 5.], [float("nan"), float("nan")]],
                      [[7., 5.], [9., 5.], [11., 5.]]])
    valid = torch.tensor([[True, True, False], [True, True, True]])
    stats = FeatureMoments(2)
    stats.update(x[:1], valid[:1]); stats.update(x[1:], valid[1:])
    mean, std = stats.finalize()
    assert stats.count == 5
    torch.testing.assert_close(mean, x[valid].mean(0))
    torch.testing.assert_close(std[0], x[valid].std(0, correction=0)[0])
    assert std[1] == 1


@pytest.mark.parametrize("pool", [False, True])
def test_attention_padding_invariance_gradients_and_serialization(pool):
    torch.manual_seed(42)
    model = AttentiveProbe(12, 7, torch.randn(12), torch.rand(12) + .1, num_heads=3, pool=pool)
    x = torch.randn(2, 5, 12, requires_grad=True)
    valid = torch.tensor([[True, True, True, False, False], [True] * 5])
    expected = model(x, valid)
    poisoned = x.detach().clone(); poisoned[~valid] = float("nan")
    torch.testing.assert_close(model(poisoned, valid), expected)
    if pool:
        assert expected.shape == (2, 7)
    else:
        assert expected.shape == (2, 5, 7)
        assert (expected[~valid] == 0).all()
    expected.square().sum().backward()
    assert (x.grad[~valid] == 0).all()
    assert model.attention.in_proj_weight.grad.abs().sum() > 0
    assert not model.feature_mean.requires_grad and not model.feature_std.requires_grad
    restored = AttentiveProbe(12, 7, torch.zeros(12), torch.ones(12), num_heads=3, pool=pool)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(x.detach(), valid), expected)


def test_intermediate_feature_matches_actual_block_and_default_unchanged():
    torch.manual_seed(0)
    model = MotionTransformer1D(in_chans=6, num_frames=5, embed_dim=12, depth=3, num_heads=3).eval()
    motion, fps = torch.randn(2, 5, 6), torch.tensor([30., 30.])
    valid = torch.tensor([[True] * 5, [True, True, True, False, False]])
    captured = []
    hooks = [block.register_forward_hook(lambda _m, _a, out: captured.append(out.detach().clone())) for block in model.blocks]
    default = model(motion, fps, valid_frames=valid)
    for h in hooks: h.remove()
    for layer in (1, 2, 3):
        selected = model(motion, fps, valid_frames=valid, feature_layer=layer)
        torch.testing.assert_close(selected, captured[layer - 1].masked_fill(~valid[..., None], 0))
    torch.testing.assert_close(default, model.norm(captured[-1]).masked_fill(~valid[..., None], 0))
    for bad in (0, 4, True, 1.5):
        with pytest.raises(ValueError): model(motion, fps, feature_layer=bad)


def test_frozen_extraction_input_stats_and_release(tmp_path):
    torch.manual_seed(2)
    model = mot_tiny_1d(in_chans=6, num_frames=5).eval()
    stats = tmp_path / "stats"; stats.mkdir()
    mean, std = np.arange(6, dtype=np.float32), np.full(6, 2., dtype=np.float32)
    np.save(stats / "mean.npy", mean); np.save(stats / "std.npy", std)
    path = tmp_path / "checkpoint.pt"
    torch.save({"config": {"meta": {"model_name": "mot_tiny_1d"}, "data": {
        "motion_dim": 6, "num_frames": 5, "root_path": str(tmp_path), "normalize": True}},
        "encoder": model.state_dict(), "target_encoder": model.state_dict(), "global_step": 17}, path)
    extractor = FrozenFrameExtractor(path, 3)
    raw = torch.randn(2, 5, 6)
    lengths, fps = torch.tensor([3, 5]), torch.tensor([30., 30.])
    valid = torch.arange(5)[None] < lengths[:, None]
    normalized = ((raw - torch.tensor(mean)) / torch.tensor(std)).masked_fill(~valid[..., None], 0)
    with torch.no_grad(): expected = model(normalized, fps, valid_frames=valid, feature_layer=3)
    raw[~valid] = float("nan")
    actual = extractor(raw, fps, lengths)
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad
    assert all(not p.requires_grad for p in extractor.encoder.parameters())
    assert extractor.metadata["checkpoint_step"] == 17
    extractor.release()
    assert extractor.encoder is None and extractor.mean is None
    with pytest.raises(RuntimeError): extractor(raw, fps, lengths)
