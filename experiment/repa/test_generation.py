"""Numerical contracts for motion shortcut generation."""
import pytest
import torch
from torch import nn
from .model import MotionShortcutTransformer, TrainingModel
from .objectives import alignment_loss, interpolate, make_targets, masked_mean, sample_motion, shortcut_target


def tiny(channels=365):
    return MotionShortcutTransformer(channels, num_frames=6, embed_dim=16, depth=2, num_heads=2, feature_layer=1)


@pytest.mark.parametrize("channels", [365, 366])
def test_model_padding_initialization_and_gradients(channels):
    model = tiny(channels)
    x = torch.randn(2, 6, channels)
    valid = torch.tensor([[1]*6, [1,1,1,0,0,0]], dtype=torch.bool)
    fps, t, d = torch.tensor([30., 60.]), torch.tensor([.1, .2]), torch.tensor([.5, .25])
    velocity, features = model(x, t, d, fps, valid, True)
    assert velocity.shape == x.shape and velocity.eq(0).all()
    nn.init.normal_(model.output_proj.weight, std=.1)
    for block in model.blocks:
        nn.init.normal_(block.modulation[-1].weight, std=.1)
    original = model(x, t, d, fps, valid)
    changed = x.clone()
    changed[1, 3:] = 10000
    torch.testing.assert_close(original, model(changed, t, d, fps, valid))
    assert original[1, 3:].eq(0).all()
    assert not torch.allclose(original, model(x, t + .3, d, fps, valid))
    assert not torch.allclose(original, model(x, t, d / 2, fps, valid))
    assert not torch.allclose(original, model(x, t, d, fps * 2, valid))
    original.square().mean().backward()
    assert model.input_proj.weight.grad.isfinite().all()
    with pytest.raises(ValueError):
        model(x, t, d, fps, torch.zeros_like(valid))


class Constant(nn.Module):
    def forward(self, x, t, d, fps, valid):
        return torch.ones_like(x).masked_fill(~valid[..., None], 0) * 2


def test_shortcut_targets_and_integration():
    x = torch.randn(8, 6, 3, requires_grad=True)
    valid = torch.ones(8, 6, dtype=torch.bool)
    valid[0, 3:] = False
    fps = torch.full((8,), 30.)
    d = torch.full((8,), .5)
    target = shortcut_target(Constant(), x, torch.zeros(8), d, fps, valid)
    assert not target.requires_grad
    torch.testing.assert_close(target[valid], torch.full_like(target[valid], 2))
    for steps in (1, 2, 128):
        output = sample_motion(Constant(), x, fps, valid, steps)
        torch.testing.assert_close(output[valid], x[valid] + 2, atol=2e-5, rtol=2e-5)
        assert output[~valid].eq(0).all()
    for steps in (0, 3, 256):
        with pytest.raises(ValueError):
            sample_motion(Constant(), x, fps, valid, steps)
    xt, target, t, d, nf = make_targets(x, fps, valid, Constant())
    assert nf == 6 and not target.requires_grad
    assert ((t + d) <= 1).all()
    torch.testing.assert_close(t[6:] / d[6:], (t[6:] / d[6:]).round())
    assert xt[~valid].eq(0).all()
    torch.testing.assert_close(interpolate(x, x * 0, torch.ones(8)), x)
    torch.testing.assert_close(interpolate(x, x * 0, torch.zeros(8)), x * 0)


def test_alignment_magnitude_patch_correspondence():
    y = torch.ones(2, 2, 4)
    valid = torch.ones(2, 2, dtype=torch.bool)
    assert alignment_loss(y * 3, y, valid, "mse") == 4
    assert alignment_loss(y * 3, y, valid, "cosine").abs() < 1e-6
    wrapped = TrainingModel(tiny(), 4)
    seen = []
    wrapped.projector.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach()))
    x = torch.randn(2, 6, 365)
    active = torch.ones(2, 6, dtype=torch.bool)
    args = (x, torch.zeros(2), torch.ones(2), torch.ones(2)*30, active)
    _, features = wrapped.generator(*args, True)
    _, projected = wrapped(*args, 1, 3)
    torch.testing.assert_close(seen[0], features[:1].reshape(1,2,3,16).mean(2))
    alignment_loss(projected, y[:1], valid[:1]).backward()
    assert wrapped.projector[0].weight.grad.abs().sum() > 0
    assert wrapped.generator.input_proj.weight.grad.abs().sum() > 0


def test_masked_loss_per_sample():
    values = torch.tensor([[[2.], [2.]], [[4.], [100.]]])
    valid = torch.tensor([[True, True], [True, False]])
    torch.testing.assert_close(masked_mean(values, valid), torch.tensor([2., 4.]))
    with pytest.raises(ValueError):
        masked_mean(values, torch.zeros_like(valid))


@pytest.mark.parametrize("patch", [False, True])
def test_teacher_preprocessing_and_freezing(tmp_path, monkeypatch, patch):
    import numpy as np
    import yaml
    from model import motion_transformer_1d as module
    from .teacher import AlignmentTeacher
    def factory(**kwargs):
        return module.MotionTransformer1D(embed_dim=16, depth=2, num_heads=2, **kwargs)
    monkeypatch.setattr(module, "mot_tiny_1d", factory)
    stats = tmp_path / "stats"
    stats.mkdir()
    np.save(stats / "mean.npy", np.ones(365, np.float32) * 2)
    np.save(stats / "std.npy", np.ones(365, np.float32) * 3)
    config = {"data": {"root_path":str(tmp_path), "stats_path":"stats", "normalize":True,
              "representation":"LocalMotionJEPARep", "motion_dim":365, "num_frames":6},
              "meta":{"model_name":"mot_tiny_1d", "patchify":patch}}
    encoder = factory(in_chans=365, num_frames=6, patchify=patch)
    checkpoint = tmp_path / "teacher.pt"
    torch.save({"target_encoder":encoder.state_dict(), "config":config}, checkpoint)
    config_path = tmp_path / "teacher.yaml"
    config_path.write_text(yaml.safe_dump(config))
    settings = {"checkpoint":str(checkpoint), "config":str(config_path)}
    teacher = AlignmentTeacher(settings, config["data"], np.ones(365), np.ones(365)*2, "cpu")
    clean = torch.randn(2,6,365)
    valid = torch.arange(6)[None] < torch.tensor([6,4])[:,None]
    fps = torch.ones(2)*30
    target, mask = teacher(clean, fps, valid)
    raw_normalized = ((clean*2+1-2)/3).masked_fill(~valid[...,None], 0)
    expected = encoder.eval()(raw_normalized, fps, valid_frames=valid, feature_layer=2)
    torch.testing.assert_close(target, expected)
    assert not target.requires_grad and not teacher.extractor.encoder.training
    assert all(not p.requires_grad for p in teacher.extractor.encoder.parameters())
    prediction = target.clone().requires_grad_()
    alignment_loss(prediction, target, mask).backward()
    assert all(p.grad is None for p in teacher.extractor.encoder.parameters())
    with pytest.raises(ValueError, match="representation"):
        AlignmentTeacher(settings, {**config["data"], "representation":"MotionJEPAMotionRep"},
                         np.zeros(366),np.ones(366),"cpu")
    with pytest.raises(ValueError, match="capacities"):
        AlignmentTeacher(settings, {**config["data"], "num_frames":9}, np.zeros(365),np.ones(365),"cpu")
    np.save(stats / "std.npy", np.ones(365)*-1)
    with pytest.raises(ValueError, match="nonnegative"):
        AlignmentTeacher(settings, config["data"], np.zeros(365),np.ones(365),"cpu")
    np.save(stats / "std.npy", np.ones(365))
    torch.save({"target_encoder":{}, "config":config}, checkpoint)
    with pytest.raises(RuntimeError):
        AlignmentTeacher(settings, config["data"], np.zeros(365),np.ones(365),"cpu")


@pytest.mark.parametrize("representation,channels", [("LocalMotionJEPARep",365), ("MotionJEPAMotionRep",366)])
def test_training_resume_evaluate_and_export(tmp_path, monkeypatch, representation, channels):
    import numpy as np
    from . import runner
    from torch.utils.data import TensorDataset
    source = torch.Generator().manual_seed(12)
    ds = TensorDataset(torch.randn(16,6,channels,generator=source),torch.ones(16,dtype=torch.long)*30,
                       torch.tensor([3,4,5,6]*4))
    ds.normalize = True
    ds.motion_dim = channels
    ds.mean = np.ones(channels, np.float32)
    ds.std = np.ones(channels, np.float32)*2
    monkeypatch.setattr(runner, "dataset", lambda config, split: ds)
    config = {
      "data":{"num_frames":6,"representation":representation},
      "model":{"embed_dim":16,"depth":2,"num_heads":2,"feature_layer":1},
      "alignment":{"weight":0,"loss":"mse"},
      "shortcut":{"fraction":.25,"base_steps":128,"weight":1.,"ema":.999},
      "training":{"seed":0,"batch_size":4,"workers":0,"iterations":4,"warmup":1,"lr":1e-4,
                  "final_lr":1e-6,"weight_decay":.01,"clip_grad":1.,"log_interval":2,
                  "checkpoint_interval":4,"bf16":False},
      "validation":{"samples":8,"interval":4}}
    runner.train(config, tmp_path/"full", device_name="cpu")
    runner.train(config, tmp_path/"split", device_name="cpu", max_steps=2)
    runner.train(config, tmp_path/"split", str(tmp_path/"split/latest.pt"), "cpu", max_steps=2)
    full = torch.load(tmp_path/"full/latest.pt",weights_only=False)
    resumed = torch.load(tmp_path/"split/latest.pt",weights_only=False)
    for key in full["model"]:
        torch.testing.assert_close(full["model"][key], resumed["model"][key], rtol=0,atol=0)
    for key in full["ema"]:
        torch.testing.assert_close(full["ema"][key], resumed["ema"][key],rtol=0,atol=0)
    runner.evaluate_checkpoint(tmp_path/"split/latest.pt", tmp_path/"eval.json","cpu")
    for steps in (1,2,128):
        destination = tmp_path/f"samples-{steps}"
        runner.sample_checkpoint(tmp_path/"split/latest.pt",destination,steps,2,4,30.,0,"cpu")
        values = np.load(destination/"motions.npy")
        assert values.shape == (2,6,channels) and np.isfinite(values).all()
        assert (values[:,4:]==0).all()
