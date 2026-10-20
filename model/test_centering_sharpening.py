"""Run with: python -m pytest model/test_centering_sharpening.py -q."""
import copy
from datetime import timedelta
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset

import experiment.online_metrics as metrics
import train
from dataset.motion_dataset import OffsetDistributedSampler
from model import MODEL_FACTORIES, PREDICTOR_FACTORIES
from model.motion_transformer_1d import MotionTransformer1D, MotionTransformerPredictor1D
from model.centering_sharpening import CenteredSharpenedLoss, centering_sharpening_config


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("name,value", [
    ("enabled", "false"), ("center_momentum", 1.0), ("center_momentum", -0.1),
    ("teacher_temperature", 0.0), ("student_temperature", -1.0),
    ("teacher_temperature", float("nan")), ("student_temperature", True),
])
def test_invalid_settings(name, value):
    with pytest.raises(ValueError):
        centering_sharpening_config({"loss": {"centering_sharpening": {"enabled": True, name: value}}})


def test_masked_cross_entropy_center_and_gradients():
    prediction = torch.tensor([[[0.2, -0.3], [1., -1.]], [[0.5, 0.2], [float("nan"), float("inf")]]], requires_grad=True)
    target = torch.tensor([[[1., 0.], [2., 0.]], [[0., 3.], [float("nan"), float("inf")]]], requires_grad=True)
    active = torch.tensor([[True, True], [True, False]])
    criterion = CenteredSharpenedLoss(2)
    criterion.center.copy_(torch.tensor([0.1, 0.2]))
    before = criterion.center.clone()
    expected_rows = []
    for p, t, mask in zip(prediction, target, active):
        probs = ((t[mask].detach() - before) / 0.06).softmax(-1)
        expected_rows.append(-(probs * (p[mask] / 0.1).log_softmax(-1)).sum(-1).mean())
    loss = criterion(prediction, target, active)
    torch.testing.assert_close(loss, torch.stack(expected_rows).mean())
    assert torch.equal(criterion.center, before)
    loss.backward()
    assert target.grad is None
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad[~active]) == 0
    criterion.update_center(target, active)
    torch.testing.assert_close(criterion.center, 0.9 * before + 0.1 * torch.tensor([1., 1.]))
    restored = CenteredSharpenedLoss(2)
    restored.load_state_dict(criterion.state_dict())
    assert torch.equal(restored.center, criterion.center)
    with pytest.raises(ValueError, match="valid token"):
        criterion(prediction, target, torch.zeros_like(active))


def test_temperature_sharpens_teacher_and_bfloat16_is_finite():
    target = torch.tensor([[[0.01, 0.02, -0.01]]])
    entropy = lambda p: -(p * p.log()).sum()
    assert entropy((target / 0.06).softmax(-1)) < entropy((target / 0.1).softmax(-1))
    prediction = torch.tensor([[[500., -500., 1.]]], dtype=torch.bfloat16, requires_grad=True)
    criterion = CenteredSharpenedLoss(3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = criterion(prediction, target.bfloat16(), torch.ones(1, 1, dtype=torch.bool))
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


class _Clips(Dataset):
    def __init__(self, **kwargs):
        pass

    def __len__(self):
        return 7

    def __getitem__(self, index):
        length = [12, 9, 7, 10, 8, 11, 6][index]
        motion = torch.sin(torch.arange(72).reshape(12, 6).float() * (index + 1) / 13)
        motion[length:] = 0
        return motion, 30.0, length


def _loader(**kwargs):
    dataset = _Clips()
    sampler = OffsetDistributedSampler(dataset, num_replicas=kwargs["world_size"],
                                       rank=kwargs["rank"], seed=kwargs["seed"], shuffle=True)
    loader = DataLoader(dataset, batch_size=kwargs["batch_size"], sampler=sampler,
                        drop_last=False, generator=torch.Generator().manual_seed(123))
    return dataset, loader, sampler


def _small_encoder(**kwargs):
    return MotionTransformer1D(**kwargs, embed_dim=16, depth=1, num_heads=4, drop_rate=0.1)


def _small_predictor(**kwargs):
    return MotionTransformerPredictor1D(**kwargs, predictor_embed_dim=16, depth=1, num_heads=4, drop_rate=0.1)


def _patch_runtime(patch):
    patch.setitem(MODEL_FACTORIES, "mot_tiny_1d", _small_encoder)
    patch.setitem(PREDICTOR_FACTORIES, "mot_predictor_tiny_1d", _small_predictor)
    patch.setattr(train, "make_motion_dataset", _loader)
    patch.setattr(metrics, "MotionDataset", _Clips)


def _config(folder, enabled):
    config = {
        "data": {"root_path": "synthetic", "meta_files": ["train.txt"], "batch_size": 2,
                 "num_frames": 12, "motion_dim": 6, "fps": 30, "drop_last": False},
        "logging": {"folder": str(folder), "write_tag": "test", "log_freq": 1,
                    "checkpoint_interval": 2, "tensorboard": False},
        "meta": {"model_name": "mot_tiny_1d", "predictor_name": "mot_predictor_tiny_1d",
                 "seed": 7, "use_bfloat16": False, "load_checkpoint": False},
        "optimization": {"iterations": 5, "warmup_iterations": 1, "start_lr": 0.0001,
                         "lr": 0.001, "final_lr": 0.00001, "weight_decay": 0.04,
                         "final_weight_decay": 0.04, "ema": [0.9, 0.99], "clip_grad": 1.0},
        "mask": {"enc_frame_mask_ratio": [0.85, 1.0], "pred_frame_mask_ratio": [0.15, 0.2],
                 "num_enc_masks": 2, "num_pred_masks": 2, "allow_overlap": False,
                 "context_selection": "all", "min_context_ratio": 0.2},
        "online_metrics": {"enabled": True, "interval": 2, "num_samples": 5, "batch_size": 2},
    }
    if enabled:
        config["loss"] = {"centering_sharpening": {"enabled": True}}
    return config


def _assert_equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_equal(x, y)
    else:
        assert a == b


def _exercise_resume(root, enabled):
    config = _config(root / "full", enabled)
    train.main(config, device="cpu")
    resumed = copy.deepcopy(config)
    resumed["logging"]["folder"] = str(root / "resumed")
    resumed["meta"].update(load_checkpoint=True, read_checkpoint=str(root / "full/test-it2.pth.tar"))
    if not enabled:
        resumed["loss"] = {"centering_sharpening": {"enabled": False}}
    train.main(resumed, device="cpu")
    a = torch.load(root / "full/test-latest.pth.tar", weights_only=False)
    b = torch.load(root / "resumed/test-latest.pth.tar", weights_only=False)
    for key in ("encoder", "predictor", "target_encoder", "optimizer", "lr_scheduler", "wd_scheduler",
                "momentum_scheduler", "mask_states", "global_step", "batch_in_pass", "data_pass"):
        _assert_equal(a[key], b[key])
    if enabled:
        _assert_equal(a["centering_sharpening"], b["centering_sharpening"])
        assert a["centering_sharpening"]["center"].abs().sum() > 0
    else:
        assert "centering_sharpening" not in a
    return config, resumed, a


@pytest.mark.parametrize("enabled", [False, True])
def test_training_exact_resume_and_legacy_disabled(tmp_path, monkeypatch, enabled):
    _patch_runtime(monkeypatch)
    config, resumed, checkpoint = _exercise_resume(tmp_path, enabled)
    if enabled:
        resumed["loss"]["centering_sharpening"]["teacher_temperature"] = 0.07
    else:
        resumed["loss"]["centering_sharpening"]["enabled"] = True
    with pytest.raises(ValueError, match="loss settings"):
        train.main(resumed, device="cpu")


def test_online_cross_entropy_is_repeatable_and_center_is_frozen(tmp_path, monkeypatch):
    _patch_runtime(monkeypatch)
    config = _config(tmp_path, True)
    encoder, predictor = train.init_mjepa_model_from_config(config, torch.device("cpu"))
    target = copy.deepcopy(encoder).requires_grad_(False)
    criterion = CenteredSharpenedLoss(16)
    criterion.center.fill_(0.15)
    center = criterion.center.clone()
    evaluator = metrics.OnlineRepresentationMetrics(config, config["online_metrics"], device="cpu",
                                                     collator=train._build_mask_collator(config, encoder.token_layout))
    rng = torch.get_rng_state().clone()
    first = train._evaluate_online_metrics(evaluator, encoder, predictor, target, None, 0, criterion)
    second = train._evaluate_online_metrics(evaluator, encoder, predictor, target, None, 0, criterion)
    assert first["heldout_jepa"] == second["heldout_jepa"]
    assert "heldout_jepa/cross_entropy" in metrics.tensorboard_metrics(first)
    assert torch.equal(center, criterion.center) and torch.equal(rng, torch.get_rng_state())
    assert encoder.training and predictor.training and target.training
    # The evaluator's cross-entropy must equal the criterion on the same batches.
    encoder.eval(); predictor.eval(); target.eval()
    total = count = 0
    for motion, fps, lengths in evaluator._loader():
        enc, pred = evaluator.collator.sample_masks(lengths.tolist())
        valid = torch.arange(12)[None] < lengths[:, None]
        with torch.no_grad():
            full = target(motion, fps, valid_frames=valid)
            y = train.repeat_mask_blocks(train.apply_index_masks(full, pred), len(motion), len(enc))
            x = predictor(encoder(motion, fps, enc, valid_frames=valid), fps, enc, pred)
            active = train.repeat_mask_blocks(train.index_mask_validity(pred), len(motion), len(enc))
            total += float(criterion(x, y, active)) * len(x)
            count += len(x)
    assert first["heldout_jepa"]["cross_entropy"] == pytest.approx(total / count)


def _distributed_worker(rank, root):
    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank))
    dist.init_process_group("gloo", init_method=f"file://{root}/rendezvous", rank=rank,
                            world_size=2, timeout=timedelta(seconds=60))
    try:
        criterion = CenteredSharpenedLoss(2)
        # Unequal valid counts must use global token sums, not a mean of rank means.
        target = torch.full((1, 3, 2), 1.0 if rank == 0 else 10.0)
        active = torch.tensor([[True, False, False]] if rank == 0 else [[True, True, True]])
        criterion.update_center(target, active)
        torch.testing.assert_close(criterion.center, torch.full((2,), 0.775))
        with pytest.MonkeyPatch.context() as patch:
            _patch_runtime(patch)
            _exercise_resume(Path(root), True)
    finally:
        dist.destroy_process_group()


def test_two_rank_center_sync_training_evaluation_and_resume(tmp_path):
    mp.spawn(_distributed_worker, args=(str(tmp_path),), nprocs=2, join=True)


def test_bfloat16_training_and_tensorboard(tmp_path, monkeypatch):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    _patch_runtime(monkeypatch)
    config = _config(tmp_path / "bf16", True)
    config["meta"]["use_bfloat16"] = True
    config["logging"]["tensorboard"] = True
    train.main(config, device="cpu")
    checkpoint = torch.load(tmp_path / "bf16/test-latest.pth.tar", weights_only=False)
    assert checkpoint["centering_sharpening"]["center"].dtype == torch.float32
    events = EventAccumulator(str(tmp_path / "bf16/tensorboard")).Reload()
    assert [event.step for event in events.Scalars("train/loss")] == [1, 2, 3, 4, 5]
    ce = events.Scalars("online_metrics/heldout_jepa/cross_entropy")
    assert [event.step for event in ce] == [0, 2, 4, 5]
    assert all(torch.isfinite(torch.tensor(event.value)) for event in ce)
    assert "online_metrics/representation/raw/rankme" in events.Tags()["scalars"]


def test_missing_center_checkpoint_fails(tmp_path, monkeypatch):
    _patch_runtime(monkeypatch)
    config = _config(tmp_path / "full", True)
    train.main(config, device="cpu")
    path = tmp_path / "missing-center.pth.tar"
    checkpoint = torch.load(tmp_path / "full/test-it2.pth.tar", weights_only=False)
    checkpoint.pop("centering_sharpening")
    torch.save(checkpoint, path)
    config["meta"].update(load_checkpoint=True, read_checkpoint=str(path))
    with pytest.raises(ValueError, match="no centering-sharpening state"):
        train.main(config, device="cpu")
