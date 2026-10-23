"""Three-frame tokenization and its configuration contract."""
import pytest
import torch
from torch import nn

from helper import (
    architecture_signature,
    architecture_signature_from_config,
    init_mjepa_encoder_from_config,
    init_mjepa_model_from_config,
    model_config_from_checkpoint,
)
from model.motion_transformer_1d import MotionTransformer1D, MotionTransformerPredictor1D


def encoder(**kwargs):
    return MotionTransformer1D(in_chans=2, num_frames=11, embed_dim=8,
                               depth=1, num_heads=2, **kwargs)


def test_patch_convolution_groups_frames_and_drops_remainder():
    model = encoder(patchify=True)
    assert isinstance(model.input_proj, nn.Conv1d)
    model.blocks = nn.ModuleList()
    model.norm = nn.Identity()
    with torch.no_grad():
        model.input_proj.weight.fill_(1)
        model.input_proj.bias.zero_()
    motion = torch.arange(22).reshape(1, 11, 2).float().requires_grad_()
    output = model(motion, torch.tensor([30.]))
    expected = motion[:, :9].reshape(1, 3, 6).sum(-1, keepdim=True).expand(-1, -1, 8)
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    assert motion.grad[:, :9].ne(0).all()
    assert motion.grad[:, 9:].eq(0).all()
    assert model.input_proj.weight.grad.abs().sum() > 0


def test_patch_validity_and_encoder_predictor_times():
    model = encoder(patchify=True).eval()
    predictor = MotionTransformerPredictor1D(
        num_frames=11, embed_dim=8, predictor_embed_dim=8, depth=1, num_heads=2,
        patchify=True).eval()
    motion = torch.randn(2, 11, 2)
    fps = torch.tensor([30., 60.])
    valid = torch.arange(11)[None] < torch.tensor([11, 7])[:, None]
    full = model(motion, fps, valid_frames=valid)
    assert full.shape == (2, 3, 8)
    assert full[1, 2].eq(0).all()
    changed = motion.clone()
    changed[1, 6:] = 1000.
    torch.testing.assert_close(model(changed, fps, valid_frames=valid), full)
    enc = [torch.tensor([[0, 2], [0, -1]])]
    pred = [torch.tensor([[1], [1]])]
    positions = {}
    model.rotary.register_forward_pre_hook(lambda module, args: positions.update(encoder=args[0].clone()))
    predictor.rotary.register_forward_pre_hook(lambda module, args: positions.update(predictor=args[0].clone()))
    context = model(motion, fps, enc, valid_frames=valid)
    output = predictor(context, fps, enc, pred)
    assert output.shape == (2, 1, 8) and output.isfinite().all()
    torch.testing.assert_close(positions["encoder"][0], torch.tensor([1., 7.]) / 30)
    torch.testing.assert_close(positions["predictor"][0], torch.tensor([1., 7., 4.]) / 30)
    torch.testing.assert_close(positions["predictor"][1, [0, 2]], torch.tensor([1., 4.]) / 60)
    output.square().mean().backward()
    assert model.input_proj.weight.grad.isfinite().all()
    with pytest.raises(ValueError, match="invalid or padded"):
        model(motion, fps, [torch.tensor([[0, 1], [0, 2]])], valid_frames=valid)


def test_default_preserves_frame_model_and_weights():
    torch.manual_seed(3)
    default = encoder().eval()
    torch.manual_seed(3)
    explicit = encoder(patchify=False).eval()
    explicit.load_state_dict(default.state_dict(), strict=True)
    motion, fps = torch.randn(2, 11, 2), torch.tensor([30., 60.])
    assert isinstance(default.input_proj, nn.Linear)
    assert default(motion, fps).shape == (2, 11, 8)
    torch.testing.assert_close(default(motion, fps), explicit(motion, fps), rtol=0, atol=0)


@pytest.mark.parametrize("patchify", [False, True])
def test_config_checkpoint_and_encoder_only_agree(patchify):
    config = {
        "data": {"num_frames": 11, "motion_dim": 2},
        "meta": {"model_name": "mot_tiny_1d", "predictor_name": "mot_predictor_tiny_1d",
                 "patchify": patchify},
    }
    enc, pred = init_mjepa_model_from_config(config, torch.device("cpu"))
    signature = architecture_signature(enc, pred, model_name="mot_tiny_1d",
                                       predictor_name="mot_predictor_tiny_1d", motion_dim=2)
    assert signature == architecture_signature_from_config(config)
    restored = model_config_from_checkpoint({"config": config, "architecture": signature})
    loaded = init_mjepa_encoder_from_config(restored, torch.device("cpu"))
    loaded.load_state_dict(enc.state_dict(), strict=True)
    assert loaded.token_layout == enc.token_layout == pred.token_layout
    assert enc.token_layout.token_num_frames == (3 if patchify else 11)


def test_patchify_rejects_too_short_input_and_non_boolean_config():
    with pytest.raises(ValueError, match="positive"):
        MotionTransformer1D(in_chans=2, num_frames=2, patchify=True)
    config = {"meta": {"patchify": "false"}}
    with pytest.raises(ValueError, match="boolean"):
        init_mjepa_model_from_config(config, torch.device("cpu"))
