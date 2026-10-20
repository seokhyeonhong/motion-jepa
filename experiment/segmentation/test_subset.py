"""Training-only subsets preserve label coverage and probe dataset behavior."""

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from experiment.segmentation.subset import select_training_subset
from experiment.segmentation import probe


class LabelledFrames(Dataset):
    split = "train"
    num_frames = 4
    motion_dim = 6
    num_classes = 3
    class_names = ("common", "medium", "rare")
    class_indices_60 = (0, 1)

    def __init__(self):
        self.records = [{"id": f"clip-{i}"} for i in range(20)]
        self.motion_reads = []
        self.label_reads = []

    def __len__(self):
        return len(self.records)

    def load_labels(self, index):
        self.label_reads.append(index)
        labels = np.zeros((2, 3), dtype=np.bool_)
        labels[0, 0] = True
        labels[1, 1] = index % 3 == 0
        labels[1, 2] = index == 19
        return labels, labels.any(axis=1)

    def __getitem__(self, index):
        self.motion_reads.append(index)
        labels, supervised = self.load_labels(index)
        motion = torch.full((4, 6), 9999.)
        motion[:2] = float(index)
        return (motion, 30, 2, np.pad(labels, ((0, 2), (0, 0))).astype(np.float32),
                np.pad(supervised, (0, 2)), self.records[index]["id"])


def test_fixed_nested_coverage_without_motion_reads_or_global_rng_changes():
    source = LabelledFrames()
    state = torch.get_rng_state().clone()
    small, metadata = select_training_subset(source, 0.2, 42)
    large, _ = select_training_subset(source, 0.5, 42)
    repeated, same = select_training_subset(source, 0.2, 42)
    assert len(small) == 4 and len(large) == 10
    assert set(small.indices) < set(large.indices)
    assert small.indices == repeated.indices and metadata == same
    assert len(set(small.indices)) == len(small)
    assert source.motion_reads == []
    assert torch.equal(state, torch.get_rng_state())
    assert 19 in small.indices
    assert min(metadata["selected_positive_windows"]) >= 1
    assert min(metadata["selected_positive_frames"]) >= 1
    assert metadata["sample_ids"] == [source.records[i]["id"] for i in small.indices]
    assert metadata["class_names"] == list(source.class_names)
    assert small.num_classes == source.num_classes and small.motion_dim == 6
    assert small.class_indices_60 == source.class_indices_60
    assert set(small.indices) != set(select_training_subset(source, 0.2, 7)[0].indices)


def test_full_fraction_preserves_existing_dataset_without_scanning():
    source = LabelledFrames()
    selected, metadata = select_training_subset(source, 1.0, 42)
    assert selected is source and metadata is None
    assert source.label_reads == source.motion_reads == []


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="fraction"):
        select_training_subset(LabelledFrames(), fraction, 42)


def test_rejects_evaluation_selection_missing_labels_and_small_budget():
    source = LabelledFrames()
    source.split = "val"
    with pytest.raises(ValueError, match="training"):
        select_training_subset(source, 0.5, 42)
    source.split = "train"
    source.records.pop()  # Removes the only rare-class clip.
    with pytest.raises(ValueError, match="rare"):
        select_training_subset(source, 0.5, 42)
    with pytest.raises(ValueError, match="coverage"):
        select_training_subset(LabelledFrames(), 0.05, 42)


def test_probe_statistics_and_training_reads_use_only_selected_clips():
    from types import SimpleNamespace
    source = LabelledFrames()
    subset, _ = select_training_subset(source, 0.2, 42)
    val = LabelledFrames()
    val.split = "val"
    datasets = {"train": subset, "val": val}
    args = SimpleNamespace(feature_source="raw", checkpoint=None, feature_layer=None,
                           precompute_batch_size=3, probe_heads=2, num_workers=0)
    cached, mean, std, metadata = probe.prepare_probe_datasets(datasets, args, torch.device("cpu"))
    expected = torch.tensor(subset.indices, dtype=torch.float32).repeat_interleave(2)
    torch.testing.assert_close(mean, expected.mean().expand(6))
    torch.testing.assert_close(std, expected.std(correction=0).expand(6))
    assert source.motion_reads == subset.indices
    assert val.motion_reads == list(range(len(val)))
    assert metadata["split_samples"] == {"train": 4, "val": 20}
    assert metadata["training_frames"] == 8
    assert cached["train"].sample_ids == [source.records[i]["id"] for i in subset.indices]


@pytest.mark.parametrize("mode,source", [("probe", "raw"), ("probe", "jepa"), ("supervised", "raw")])
def test_training_subset_manifest_evaluation_and_resume(tmp_path, monkeypatch, mode, source):
    import json
    from experiment.segmentation import train
    torch.set_num_threads(1)
    root = tmp_path / "data"
    root.mkdir()
    (root / "stats").mkdir()
    (root / "meta.json").write_text('{"motion_dim":6,"num_frames":4,"fps":30}')
    for name in ("index.json", "train.txt", "val.txt", "test.txt"):
        (root / name).write_text("fixture")
    np.save(root / "stats/mean.npy", np.zeros(6))
    np.save(root / "stats/std.npy", np.ones(6))
    constructed = []

    def factory(root, split, **kwargs):
        dataset = LabelledFrames()
        dataset.split = split
        if split == "test":
            dataset.records = []
        constructed.append(dataset)
        return dataset

    class Extractor:
        feature_dim = 6
        num_frames = 4
        metadata = {"checkpoint_sha256": "fixture"}
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, motion, fps, lengths):
            return motion * 2
        def release(self):
            pass

    monkeypatch.setattr(train, "BabelSegmentationDataset", factory)
    monkeypatch.setattr(probe, "FrozenFrameExtractor", Extractor)
    arguments = ["--dataset-root", str(root), "--output-root", str(tmp_path / "run"),
                 "--mode", mode, "--feature-source", source, "--model-size", "tiny",
                 "--train-fraction", "0.2", "--seed", "42", "--device", "cpu",
                 "--iterations", "2", "--warmup-iterations", "0", "--eval-interval", "1", "--checkpoint-interval", "1", "--batch-size", "2",
                 "--num-workers", "0", "--probe-heads", "2"]
    if mode == "probe" and source == "jepa":
        arguments += ["--checkpoint", "fixture", "--feature-layer", "3"]
    args = train.build_parser().parse_args(arguments)
    summary = train.run(args)
    saved = json.loads((args.output_root / "train-subset.json").read_text())
    assert saved == summary["signature"]["train_subset"]
    assert len(saved["sample_ids"]) == 4
    assert min(saved["selected_positive_windows"]) >= 1
    assert summary["train_fraction"] == 0.2
    assert summary["split_counts"] == {"train": 4, "val": 20, "test": 0}
    assert summary["test"] is None
    assert len(constructed[0]) == 20
    if mode == "probe":
        assert len(constructed[0].motion_reads) == 4
        assert summary["signature"]["features"]["training_frames"] == 8
    checkpoint = torch.load(args.output_root / "latest.pth.tar", weights_only=False)
    assert checkpoint["history"][0]["val"]["supervised_frames"] == 28
    args.resume = True
    resumed = train.run(args)
    assert resumed["best_val_frame_map"] == summary["best_val_frame_map"]
    args.train_fraction = 0.5
    with pytest.raises(ValueError, match="Resume training subset"):
        train.run(args)
    args.train_fraction = 0.2
    args.seed = 7
    with pytest.raises(ValueError, match="Resume training subset"):
        train.run(args)


def test_label_only_reader_checks_integrity_and_masks_without_motion(tmp_path):
    from dataset.babel_segmentation import BabelSegmentationDataset, file_sha256
    dataset = object.__new__(BabelSegmentationDataset)
    dataset.root = tmp_path
    dataset.num_classes = 3
    labels = np.array([[1, 0, 0], [0, 0, 1]], dtype=np.bool_)
    path = tmp_path / "labels.npz"
    np.savez(path, labels=labels, supervision=labels.any(axis=1))
    dataset.records = [{"labels_path": "labels.npz", "length": 2,
                        "labels_sha256": file_sha256(path)}]
    actual, supervised = dataset.load_labels(0)
    np.testing.assert_array_equal(actual, labels)
    assert supervised.all()  # No motion entries or files exist in this fixture.
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="label hash mismatch"):
        dataset.load_labels(0)
    np.savez(path, labels=labels, supervision=np.zeros(2, dtype=np.bool_))
    dataset.records[0]["labels_sha256"] = file_sha256(path)
    with pytest.raises(ValueError, match="supervision disagrees"):
        dataset.load_labels(0)
