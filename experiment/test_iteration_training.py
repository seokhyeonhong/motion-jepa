"""Exercise fixed update budgets, evaluation intervals, and checkpoint resume."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiment.tmr import train as tmr, model as tmr_model
from experiment.tmr.dataset import PreparedPairDataset
from experiment.tmr.test_probing import fixtures, TextBank
from experiment.segmentation import train as segmentation
from experiment.segmentation.test_probe import Frames


def tmr_fixture(size):
    datasets, provenance = fixtures()
    source = datasets["train"].raw_dataset
    values, entries = list(source.values), list(source.entries)
    source.values, source.entries = [], []
    records = []
    for index in range(size):
        original = index % 3
        sample_id = f"train-{index}"
        entry = SimpleNamespace(sample_id=sample_id, path=source.root_path / f"{sample_id}.npy",
                                length=entries[original].length, fps=30)
        source.values.append(values[original].copy())
        source.entries.append(entry)
        captions = [f"caption-{(original + i) % 3}" for i in range(3)]
        records.append(dict(sample_id=sample_id, motion_path=entry.path.name, length=entry.length,
                            fps=30, caption_id=captions[0], caption_ids=captions,
                            source_id=sample_id, start_frame=0, end_frame=entry.length))
    datasets["train"] = PreparedPairDataset(records, TextBank(), raw_dataset=source, sample_captions=True)
    return datasets, provenance


class ManyFrames(Frames):
    def __init__(self, split, size):
        super().__init__(split)
        if split == "train":
            self.count = size

    def __getitem__(self, index):
        *values, _ = super().__getitem__(index % 3)
        return (*values, f"{self.split}-{index}")


def setup_trainer(task, tmp_path, monkeypatch, size=7):
    torch.set_num_threads(1)
    if task == "tmr":
        module = tmr
        monkeypatch.setitem(tmr_model.MODEL_SPECS, "base", {"embed_dim": 12, "depth": 2, "num_heads": 3})
        monkeypatch.setattr(tmr, "load_prepared_datasets", lambda **kwargs: tmr_fixture(size))
        extras = ["--probe-heads", "3", "--no-tensorboard", "--no-use-bfloat16"]
    else:
        module = segmentation
        root = tmp_path / "dataset"
        root.mkdir(exist_ok=True)
        (root / "stats").mkdir(exist_ok=True)
        (root / "meta.json").write_text('{"motion_dim":6,"num_frames":4,"fps":30}')
        for name in ("index.json", "train.txt", "val.txt", "test.txt"):
            (root / name).write_text("fixture")
        np.save(root / "stats/mean.npy", np.zeros(6))
        np.save(root / "stats/std.npy", np.ones(6))
        monkeypatch.setattr(segmentation, "BabelSegmentationDataset",
                            lambda root, split, **kwargs: ManyFrames(split, size))
        extras = ["--dataset-root", str(root), "--probe-heads", "2"]
    args = module.build_parser().parse_args([
        "--mode", "probe", "--feature-source", "raw", "--device", "cpu", "--num-workers", "0",
        "--iterations", "7", "--warmup-iterations", "2", "--batch-size", "2",
        "--eval-batch-size", "2", "--eval-interval", "3", "--checkpoint-interval", "2",
        "--log-freq", "2", *extras])
    return module, args


@pytest.mark.parametrize("task", ["tmr", "segmentation"])
@pytest.mark.parametrize("stop", [2, 4])
def test_resume_continues_iteration_budget_and_schedule(task, stop, tmp_path, monkeypatch):
    module, args = setup_trainer(task, tmp_path, monkeypatch)
    args.output_root = tmp_path / "full"
    module.run(args)
    baseline = torch.load(args.output_root / "latest.pth.tar", weights_only=False)
    args.output_root = tmp_path / "resumed"
    save = module._atomic_torch_save

    def interrupt(payload, path):
        save(payload, path)
        if path.name == "latest.pth.tar" and payload["global_step"] == stop:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(module, "_atomic_torch_save", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        module.run(args)
    saved = torch.load(args.output_root / "latest.pth.tar", weights_only=False)
    assert saved["global_step"] == stop
    assert "format_version" not in saved
    monkeypatch.setattr(module, "_atomic_torch_save", save)
    args.resume = True
    actual = module.run(args)
    resumed = torch.load(args.output_root / "latest.pth.tar", weights_only=False)
    assert resumed["scheduler"] == baseline["scheduler"]
    assert resumed["history"][:len(saved["history"])] == saved["history"]
    assert [(row["iteration"], row["lr"]) for row in resumed["history"]] == [
        (row["iteration"], row["lr"]) for row in baseline["history"]]
    assert actual["iterations_completed"] == 7
    assert actual["samples_seen"] == 14
    assert "format_version" not in resumed
    assert [row["iteration"] for row in resumed["history"]] == [3, 6, 7]
    assert resumed["history"][-1]["lr"] == args.final_lr
    assert all(int(state["step"]) == 7 for state in resumed["optimizer"]["state"].values())
    args.iterations += 1
    with pytest.raises(ValueError, match="(config|settings)"):
        module.run(args)


@pytest.mark.parametrize("task", ["tmr", "segmentation"])
def test_dataset_size_does_not_change_update_budget_or_lr(task, tmp_path, monkeypatch):
    histories = []
    for size in (7, 15):
        module, args = setup_trainer(task, tmp_path, monkeypatch, size)
        args.output_root = tmp_path / str(size)
        result = module.run(args)
        saved = torch.load(args.output_root / "latest.pth.tar", weights_only=False)
        histories.append([(row["iteration"], row["lr"]) for row in saved["history"]])
        assert result["global_step"] == 7
        assert result["split_counts"]["train"] == size
        assert result["split_counts"]["val"] == (3 if task == "tmr" else 2)
    assert histories[0] == histories[1]


@pytest.mark.parametrize("module", [tmr, segmentation])
def test_epoch_cli_is_rejected(module):
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(["--epochs", "100"])


@pytest.mark.parametrize("task", ["tmr", "segmentation"])
@pytest.mark.parametrize("option,value", [("iterations", 0), ("warmup_iterations", 7),
                                          ("eval_interval", 0)])
def test_invalid_iteration_settings(task, option, value, tmp_path, monkeypatch):
    module, args = setup_trainer(task, tmp_path, monkeypatch)
    args.output_root = tmp_path / "invalid"
    setattr(args, option, value)
    with pytest.raises(ValueError):
        module.run(args)
    assert not args.output_root.exists()
