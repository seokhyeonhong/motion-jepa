"""Segmentation probe tests for cache boundaries and the real training path."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from experiment.segmentation import probe, train
from experiment.segmentation.model import MotionSegmentationProbe


class Frames(Dataset):
    num_frames = 4
    motion_dim = 6
    num_classes = 2
    class_indices_60 = (0,)

    def __init__(self, split):
        self.split = split
        self.calls = 0
        self.count = {"train": 3, "val": 2, "test": 0}[split]
        self.offset = 100 if split == "val" else 0

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        self.calls += 1
        length = index + 1
        motion = torch.full((self.num_frames, self.motion_dim), 9999.)
        motion[:length] = (torch.arange(length * self.motion_dim).reshape(length, -1)
                           + index + self.offset).float()
        labels = torch.zeros(self.num_frames, self.num_classes)
        labels[0, index % 2] = 1
        supervised = labels.any(dim=-1)
        return motion, 30, length, labels, supervised, f"{self.split}-{index}"


def cache_args(**kwargs):
    result = dict(feature_source="raw", checkpoint=None, feature_layer=None,
                  encoder_key="target_encoder", precompute_batch_size=2,
                  probe_heads=2, num_workers=0)
    result.update(kwargs)
    return SimpleNamespace(**result)


def test_ram_cache_train_only_valid_moments_and_alignment(tmp_path):
    datasets = {split: Frames(split) for split in ("train", "val", "test")}
    expected = torch.cat([datasets["train"][i][0][:i+1] for i in range(3)])
    for data in datasets.values():
        data.calls = 0
    cached, mean, std, metadata = probe.prepare_probe_datasets(
        datasets, cache_args(), torch.device("cpu"),
    )
    torch.testing.assert_close(mean, expected.mean(0))
    torch.testing.assert_close(std, expected.std(0, correction=0))
    assert metadata["training_frames"] == 6
    assert metadata["split_samples"] == {"train": 3, "val": 2, "test": 0}
    assert metadata["storage"] == "cpu_ram_only"
    assert cached["val"].sample_ids == ["val-0", "val-1"]
    for split, data in cached.items():
        assert datasets[split].calls == len(data)
        assert data.features.device.type == "cpu"
        assert data.labels.dtype == torch.bool
        for index in range(len(data)):
            features, fps, length, labels, supervised, sample_id = data[index]
            assert not features[int(length):].any()
            assert torch.equal(supervised, labels.any(dim=-1))
            assert sample_id == f"{split}-{index}"
            assert int(fps) == 30
        assert datasets[split].calls == len(data), "Cached access must not reread original files"
    assert list(tmp_path.iterdir()) == []


def test_frozen_extractor_is_inference_only_and_released(monkeypatch):
    extractors = []

    class Extractor:
        feature_dim = 6
        num_frames = 4
        metadata = {"checkpoint_sha256": "fixture"}

        def __init__(self, checkpoint, layer, **kwargs):
            self.released = False
            self.layer = layer
            extractors.append(self)

        def __call__(self, motion, fps, lengths):
            assert torch.is_inference_mode_enabled()
            return motion * 2

        def release(self):
            self.released = True

    monkeypatch.setattr(probe, "FrozenFrameExtractor", Extractor)
    datasets = {split: Frames(split) for split in ("train", "val", "test")}
    cached, _, _, metadata = probe.prepare_probe_datasets(
        datasets, cache_args(feature_source="jepa", checkpoint="fixture", feature_layer=3),
        torch.device("cpu"),
    )
    assert extractors[0].released
    assert metadata["layer"] == 3
    assert metadata["checkpoint_sha256"] == "fixture"
    assert not cached["train"].features.requires_grad

    def fail(*args):
        raise RuntimeError("extraction failed")

    monkeypatch.setattr(Extractor, "__call__", fail)
    with pytest.raises(RuntimeError, match="extraction failed"):
        probe.prepare_probe_datasets(
            datasets, cache_args(feature_source="jepa", checkpoint="fixture", feature_layer=3),
            torch.device("cpu"),
        )
    assert extractors[-1].released


def test_frame_probe_preserves_frame_outputs_and_masks_padding():
    torch.manual_seed(1)
    model = MotionSegmentationProbe(6, 2, 4, torch.zeros(6), torch.ones(6), num_heads=2)
    features = torch.randn(2, 4, 6)
    valid = torch.tensor([[True, True, False, False], [True, True, True, True]])
    result = model(features, valid, torch.tensor([30, 30]))
    altered = features.clone()
    altered[0, 2:] = 100000
    torch.testing.assert_close(result, model(altered, valid, torch.tensor([30, 30])))
    assert result.shape == (2, 4, 2)
    assert not result[0, 2:].any()
    assert model.parameter_counts()["encoder"] == 0


def test_real_probe_training_checkpoint_and_resume(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "stats").mkdir()
    (root / "meta.json").write_text('{"motion_dim":6,"num_frames":4,"fps":30}')
    for name in ("index.json", "train.txt", "val.txt", "test.txt"):
        (root / name).write_text("fixture")
    np.save(root / "stats/mean.npy", np.zeros(6))
    np.save(root / "stats/std.npy", np.ones(6))
    constructed = []

    def dataset_factory(root, split, **kwargs):
        assert kwargs["normalize"] is False
        result = Frames(split)
        constructed.append(result)
        return result

    monkeypatch.setattr(train, "BabelSegmentationDataset", dataset_factory)
    output = tmp_path / "run"
    args = train.build_parser().parse_args([
        "--dataset-root", str(root), "--output-root", str(output), "--mode", "probe",
        "--feature-source", "raw", "--device", "cpu", "--iterations", "2", "--warmup-iterations", "0",
        "--eval-interval", "1", "--checkpoint-interval", "1", "--batch-size", "2", "--num-workers", "0", "--probe-heads", "2",
    ])
    summary = train.run(args)
    assert summary["iterations_completed"] == 2
    assert summary["mode"] == "probe"
    assert summary["input_source"] == "raw"
    assert summary["test"] is None
    assert summary["parameters"]["encoder"] == 0
    assert [dataset.calls for dataset in constructed] == [3, 2, 0]
    checkpoint = torch.load(output / "latest.pth.tar", weights_only=False)
    assert "features" not in checkpoint
    assert checkpoint["signature"]["features"]["training_frames"] == 6
    assert checkpoint["history"][0]["train"]["supervised_frames"] == 2
    assert checkpoint["history"][0]["val"]["supervised_frames"] == 2
    assert all(tensor.numel() < 1000 for tensor in checkpoint["model"].values())
    args.resume = True
    resumed = train.run(args)
    assert resumed["best_step"] == summary["best_step"]
    assert resumed["best_val_frame_map"] == summary["best_val_frame_map"]


def test_existing_supervised_default():
    args = train.build_parser().parse_args([])
    assert args.mode == "supervised"
    assert args.model_size == "base"
    assert args.feature_layer is None
    assert args.checkpoint is None
