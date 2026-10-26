import pytest
import torch

from .model import MotionSegmentationTransformer
from .train import build_parser, _validate_motion_depth


@pytest.mark.parametrize("depth", [3, 6, 9, 12])
def test_base_depth_forward_backward_and_checkpoint(depth):
    torch.set_num_threads(2)
    model = MotionSegmentationTransformer(num_frames=4, motion_depth=depth)
    assert len(model.encoder.blocks) == depth
    assert model.config["depth"] == depth
    assert model.config["embed_dim"] == 384
    assert model.config["num_heads"] == 6
    valid = torch.tensor([[True, True, False, False]])
    out = model(torch.randn(1, 4, 366), valid, torch.tensor([30.]))
    assert out.shape == (1, 4, 120)
    assert torch.count_nonzero(out[:, 2:]) == 0
    out.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    restored = MotionSegmentationTransformer(num_frames=4, motion_depth=depth)
    restored.load_state_dict(model.state_dict(), strict=True)


def test_default_base_matches_explicit_depth():
    torch.manual_seed(42)
    default = MotionSegmentationTransformer(num_frames=4)
    torch.manual_seed(42)
    explicit = MotionSegmentationTransformer(num_frames=4, motion_depth=12)
    assert default.config == explicit.config
    assert all(torch.equal(v, explicit.state_dict()[k]) for k, v in default.state_dict().items())


@pytest.mark.parametrize("depth", [0, -1, True, 1.5])
def test_invalid_depth(depth):
    with pytest.raises(ValueError, match="positive integer"):
        MotionSegmentationTransformer(motion_depth=depth)


def test_cli_and_probe_rejection():
    args = build_parser().parse_args(["--motion-depth", "6"])
    _validate_motion_depth(args)
    assert args.motion_depth == 6
    args.mode = "probe"
    with pytest.raises(ValueError, match="supervised"):
        _validate_motion_depth(args)
