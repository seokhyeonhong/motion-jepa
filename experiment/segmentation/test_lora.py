import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from . import train, probe
from .lora import LoRALinear, LoRASegmentation, prepare_lora_model
from .test_concat_probe import checkpoint, PatchFrames, args_for


def lora_args(checkpoint):
    args = args_for(checkpoint)
    args.lora_rank, args.lora_alpha = 8, None
    args.feature_layers = None
    args.feature_layer = None
    return args


def test_zero_init_and_only_adapters_update():
    base = nn.Linear(12, 36)
    layer = LoRALinear(base)
    x = torch.randn(4, 12)
    torch.testing.assert_close(layer(x), base(x), rtol=0, atol=0)
    before = base.weight.detach().clone()
    opt = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=.01)
    layer(x).square().mean().backward()
    assert base.weight.grad is None and layer.lora_B.grad.abs().sum() > 0
    opt.step()
    torch.testing.assert_close(base.weight, before, rtol=0, atol=0)
    assert layer.lora_B.abs().sum() > 0


def test_online_gradients_masking_and_roundtrip(checkpoint):
    args = lora_args(checkpoint)
    model = prepare_lora_model(PatchFrames("train"), args, torch.device("cpu"))
    frozen = probe.FrozenPatchLayersExtractor(checkpoint, [4])
    motion = torch.randn(2, 6, 6)
    lengths = torch.tensor([4, 6])
    valid = torch.arange(6)[None] < lengths[:, None]
    fps = torch.tensor([30., 30.])
    torch.testing.assert_close(model.extract(motion, valid, fps),
                               frozen.encoder.norm(frozen(motion, fps, lengths)), rtol=0, atol=0)
    model.train()
    assert not model.encoder.training and model.head.training
    output = model(motion, valid, fps)
    assert output.shape == (2, 6, 2)
    output[valid].square().mean().backward()
    grads = [(n, p.grad) for n, p in model.encoder.named_parameters()]
    assert all(g is None for n, g in grads if "lora_" not in n)
    assert any(g is not None and g.abs().sum() > 0 for n, g in grads if "lora_B" in n)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    motion[~valid] = float("nan")
    torch.testing.assert_close(model(motion, valid, fps), output)
    clone = prepare_lora_model(PatchFrames("train"), args, torch.device("cpu"), model.state_dict())
    torch.testing.assert_close(clone(motion, valid, fps), output)
    assert all(not b._forward_hooks for b in model.encoder.blocks)


def test_lora_training_resume_and_signature(tmp_path, monkeypatch, checkpoint):
    root = tmp_path / "data"
    (root / "stats").mkdir(parents=True)
    (root / "meta.json").write_text(json.dumps(dict(motion_dim=6, num_frames=6, fps=30)))
    for name in ("index.json", "train.txt", "val.txt", "test.txt"):
        (root / name).write_text("fixture")
    np.save(root / "stats/mean.npy", np.zeros(6))
    np.save(root / "stats/std.npy", np.ones(6))
    monkeypatch.setattr(train, "BabelSegmentationDataset", lambda root, split, **kw: PatchFrames(split))
    args = train.build_parser().parse_args([
        "--mode", "lora", "--probe-architecture", "patch-transformer",
        "--checkpoint", str(checkpoint),
        "--dataset-root", str(root), "--output-root", str(tmp_path / "run"),
        "--device", "cpu", "--iterations", "2", "--warmup-iterations", "0",
        "--eval-interval", "1", "--checkpoint-interval", "1",
        "--batch-size", "2", "--num-workers", "0"])
    assert args.lora_rank == 8
    result = train.run(args)
    assert result["mode"] == "lora" and result["input_source"] == "jepa"
    assert result["global_step"] == 2
    args.resume = True
    assert train.run(args)["best_step"] == result["best_step"]
    args.lora_alpha = 16
    with pytest.raises(ValueError, match="Resume dataset"):
        train.run(args)


@pytest.mark.parametrize("rank", [0, -1, True, 1.5])
def test_invalid_rank(rank):
    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(3, 3), rank)


def test_frame_checkpoint_and_direct_linear_head(tmp_path, monkeypatch):
    from experiment import probing
    from model.motion_transformer_1d import MotionTransformer1D
    def factory(in_chans, num_frames, patchify):
        return MotionTransformer1D(in_chans=in_chans, num_frames=num_frames,
                                   embed_dim=12, depth=3, num_heads=2, patchify=patchify)
    monkeypatch.setattr(probing.motion_transformer_1d, "mot_lora_frame_fixture", factory, raising=False)
    path = tmp_path / "frame.pth"
    torch.save(dict(config=dict(data=dict(motion_dim=6, num_frames=6, normalize=False),
                               meta=dict(model_name="mot_lora_frame_fixture", patchify=False)),
                    target_encoder=factory(6, 6, False).state_dict()), path)
    args = lora_args(path)
    args.probe_architecture = "attentive"
    args.feature_layers, args.feature_layer = None, 3
    dataset = PatchFrames("train")
    model = prepare_lora_model(dataset, args, torch.device("cpu"))
    features = []
    for motion, fps, length, *_ in dataset:
        valid = torch.arange(6)[None] < length
        features.append(model.extract(motion[None], valid, torch.tensor([fps]))[valid])
    features = torch.cat(features)
    assert isinstance(model.head, nn.Linear)
    assert model.head.in_features == 12 and model.head.out_features == 2
    motion, fps, length, *_ = dataset[0]
    output = model(motion[None], torch.arange(6)[None] < length, torch.tensor([fps]))
    output.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() for n,p in model.named_parameters() if "lora_B" in n)


def test_reject_intermediate_concat(checkpoint):
    args = lora_args(checkpoint)
    args.feature_layers = [1, 3]
    with pytest.raises(ValueError, match="final JEPA output"):
        LoRASegmentation(args, 6, 2)
