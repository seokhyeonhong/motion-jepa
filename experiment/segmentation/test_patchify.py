import io
import pytest
import torch
from .model import MotionSegmentationTransformer
from .train import build_parser, _validate_motion_depth

torch.set_num_threads(2)

@pytest.mark.parametrize("dim", [365, 366])
@pytest.mark.parametrize("frames", [1, 5, 150])
def test_patch_padding_tail_gradients_and_checkpoint(dim, frames):
    model = MotionSegmentationTransformer("tiny", input_dim=dim, num_frames=frames,
                                         motion_depth=1, patchify=True).eval()
    valid = torch.ones(2, frames, dtype=torch.bool)
    if frames > 1:
        valid[1, -1] = False
    x = torch.randn(2, frames, dim, requires_grad=True)
    out = model(x, valid, torch.full((2,), 30.))
    assert out.shape == (2, frames, 120)
    assert torch.isfinite(out).all()
    changed = x.detach().clone()
    changed[~valid] = 1e6
    torch.testing.assert_close(model(changed, valid, torch.full((2,), 30.)), out)
    assert torch.count_nonzero(out[~valid]) == 0
    out[0, -1].square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[0, -1].abs().sum() > 0
    assert x.grad[~valid].abs().sum() == 0
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = MotionSegmentationTransformer("tiny", input_dim=dim, num_frames=frames,
                                            motion_depth=1, patchify=True).eval()
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    torch.testing.assert_close(restored(changed, valid, torch.full((2,), 30.)), out)

def test_frame_order_and_fully_padded_patch_mask():
    model = MotionSegmentationTransformer("tiny", input_dim=365, num_frames=8,
                                         motion_depth=1, patchify=True)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.copy_(torch.arange(360))
    valid = torch.tensor([[True, True, True, True, False, False, False, False]])
    observed = {}
    def capture(module, args, kwargs):
        observed["mask"] = kwargs["valid_frames"]
    hook = model.encoder.register_forward_pre_hook(capture, with_kwargs=True)
    out = model(torch.randn(1, 8, 365), valid, torch.tensor([30.]))
    hook.remove()
    assert observed["mask"].tolist() == [[True]*6 + [False]*3]
    for frame in range(4):
        torch.testing.assert_close(out[0, frame], torch.arange(120).float() + (frame % 3)*120)
    assert torch.count_nonzero(out[0, 4:]) == 0

def test_default_compatibility_and_cli():
    torch.manual_seed(42)
    old = MotionSegmentationTransformer("tiny")
    torch.manual_seed(42)
    explicit = MotionSegmentationTransformer("tiny", patchify=False)
    assert old.config == explicit.config
    assert all(torch.equal(v, explicit.state_dict()[k]) for k,v in old.state_dict().items())
    args = build_parser().parse_args(["--patchify", "--motion-depth", "6"])
    _validate_motion_depth(args)
    assert args.patchify
    args.mode = "probe"
    with pytest.raises(ValueError, match="supervised"):
        _validate_motion_depth(args)
