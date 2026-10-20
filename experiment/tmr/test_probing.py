"""TMR probing contracts: train-only statistics, RAM lifecycle, and checkpoint reuse."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiment.tmr.dataset import PreparedPairDataset
from experiment.tmr import model as model_module
from experiment.tmr import probing, train


class RawDataset:
    def __init__(self, split, offset, width=6):
        self.root_path = __import__("pathlib").Path("/synthetic")
        self.normalize = True
        self.values = []
        self.entries = []
        for index, length in enumerate((2, 4, 3)):
            sample_id = f"{split}-{index}"
            values = np.zeros((4, width), dtype=np.float32)
            values[:length] = np.arange(length * width).reshape(length, width) + offset + index
            self.values.append(values)
            self.entries.append(SimpleNamespace(
                sample_id=sample_id, path=self.root_path / f"{sample_id}.npy", length=length, fps=30,
            ))

    def __getitem__(self, index):
        values = self.values[index]
        return values.copy(), 30, self.entries[index].length


class TextBank:
    def __init__(self):
        self.key_to_row = {f"caption-{i}": i for i in range(3)}

    def __getitem__(self, key):
        value = torch.zeros(1, 8)
        value[0, self.key_to_row[key]] = 1
        return value


def fixtures():
    datasets = {}
    for split, offset in (("train", 0), ("val", 1000), ("test", 2000)):
        base = RawDataset(split, offset)
        records = []
        for index, entry in enumerate(base.entries):
            records.append({
                "sample_id": entry.sample_id, "length": entry.length, "fps": 30,
                "motion_path": entry.path.name, "caption_id": f"caption-{index}",
                "caption_ids": [f"caption-{index}"], "source_id": entry.sample_id,
                "start_frame": 0, "end_frame": entry.length,
            })
        datasets[split] = PreparedPairDataset(
            records, TextBank(), raw_dataset=base, sample_captions=split == "train",
        )
    return datasets, {"motion_dim": 6, "text_dim": 8, "provenance": {"num_frames": 4, "fps": 30}}


def configuration(source="raw"):
    return {"feature_source": source, "checkpoint": "/synthetic/model.pth", "feature_layer": 3, "encoder_key": "target_encoder"}


def test_ram_cache_uses_actual_valid_train_frames_and_releases_raw_datasets():
    datasets, metadata = fixtures()
    expected = torch.from_numpy(np.concatenate([
        values[:entry.length] for values, entry in zip(datasets["train"].raw_dataset.values, datasets["train"].raw_dataset.entries)
    ]))
    datasets, provenance, mean, std = probing.prepare_probe_datasets(
        datasets, metadata, configuration(), torch.device("cpu"), batch_size=2,
    )
    torch.testing.assert_close(mean.float(), expected.mean(0))
    torch.testing.assert_close(std.float(), expected.std(0, correction=0))
    for dataset in datasets.values():
        assert dataset.raw_dataset is None and dataset.raw_rows is None
        assert dataset.motion_bank.values.device.type == "cpu"
        assert dataset.motion_bank.values.dtype == torch.float32
        assert dataset[1]["motion_tokens"].shape == (4, 6)
    assert provenance["motion_dim"] == 6
    assert datasets["val"][0]["motion_tokens"].mean() > mean.mean() + 900


def test_jepa_features_are_released_and_only_train_updates_moments(monkeypatch):
    instances = []
    class Extractor:
        feature_dim = 6
        metadata = {"layer": 3}
        def __init__(self, *args, **kwargs):
            self.released = False
            instances.append(self)
        def __call__(self, motion, fps, lengths):
            return motion * 2
        def release(self):
            self.released = True
    monkeypatch.setattr(probing, "FrozenFrameExtractor", Extractor)
    datasets, metadata = fixtures()
    datasets, _, mean, _ = probing.prepare_probe_datasets(
        datasets, metadata, configuration("jepa"), torch.device("cpu"), batch_size=2,
    )
    assert instances[0].released
    torch.testing.assert_close(datasets["train"][0]["motion_tokens"][0], torch.arange(6).float() * 2)
    assert mean.max() < 100


def test_probe_changes_only_motion_architecture_and_round_trips(monkeypatch):
    monkeypatch.setitem(model_module.MODEL_SPECS, "base", {"embed_dim": 12, "depth": 2, "num_heads": 3})
    torch.manual_seed(7)
    supervised = model_module.TextMotionAlignment(model_module.AlignmentConfig(8, 6, 4, 30))
    torch.manual_seed(7)
    config = model_module.AlignmentConfig(8, 6, 4, 30, motion_encoder="attentive_probe", probe_heads=3)
    model = model_module.TextMotionAlignment(config, motion_mean=torch.arange(6).float(), motion_std=torch.ones(6) * 2)
    for key, value in supervised.text_encoder.state_dict().items():
        torch.testing.assert_close(model.text_encoder.state_dict()[key], value, rtol=0, atol=0)
    assert not any("backbone" in key for key in model.motion_encoder.state_dict())
    tokens = torch.randn(2, 4, 6)
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    output = model.encode_motion(tokens, mask)
    assert output.shape == (2, 256)
    torch.testing.assert_close(output.norm(dim=-1), torch.ones(2))
    restored = model_module.TextMotionAlignment(config)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored.encode_motion(tokens, mask), output)


def test_probe_training_checkpoint_and_completed_resume(tmp_path, monkeypatch):
    monkeypatch.setitem(model_module.MODEL_SPECS, "base", {"embed_dim": 12, "depth": 2, "num_heads": 3})
    monkeypatch.setattr(train, "load_prepared_datasets", lambda **kwargs: fixtures())
    args = train.build_parser().parse_args([
        "--mode", "probe", "--feature-source", "raw", "--probe-heads", "3",
        "--output-root", str(tmp_path), "--device", "cpu", "--epochs", "1", "--warmup-epochs", "0",
        "--batch-size", "3", "--eval-batch-size", "3", "--num-workers", "0", "--no-tensorboard",
    ])
    result = train.run(args)
    assert result["epochs_completed"] == 1 and result["global_step"] == 1
    checkpoint = train.load_alignment_checkpoint(tmp_path / "best.pth.tar")
    assert checkpoint["config"]["model"]["motion_encoder"] == "attentive_probe"
    assert not any("motion_encoder.backbone" in key for key in checkpoint["model"])
    assert checkpoint["config"]["probe"]["cache"] == "anonymous_cpu_float32"
    assert not any(path.suffix in {".npy", ".npz"} for path in tmp_path.rglob("*"))
    args.resume = True
    resumed = train.run(args)
    assert resumed["test"] == result["test"]
    assert resumed["global_step"] == 1
