"""Check suite forwarding and fraction-specific report comparisons without jobs."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from experiment import probe_suite, probing_report


@pytest.mark.parametrize("fraction,suffix", [(1.0, ""), (0.1, "-frac0.1"), (0.01, "-frac0.01")])
def test_suite_forwards_fraction_and_separates_outputs(tmp_path, monkeypatch, fraction, suffix):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fixture")
    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(probe_suite.sys, "argv", ["probe_suite", "--checkpoint", str(checkpoint),
        "--name", "fixture", "--task", "segmentation", "--runs", "raw", "3",
        "--train-fraction", str(fraction), "--report-start", "6"])

    def run(command, **kwargs):
        calls.append(command)
        if command[2] == "experiment.segmentation":
            assert command[command.index("--train-fraction") + 1] == str(fraction)
            assert command[command.index("--seed") + 1] == "42"
            output = Path(command[command.index("--output-root") + 1])
            assert output.parents[1].name == "fixture" + suffix
            output.mkdir(parents=True)
            (output / "summary.json").write_text("{}")
        else:
            report = Path(command[command.index("--report-dir") + 1])
            assert report.name == "006-segmentation-probing" + suffix
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(probe_suite.subprocess, "run", run)
    probe_suite.main()
    assert len(calls) == 4


def test_report_requires_matching_fraction_and_sample_ids(tmp_path):
    def saved(name, fraction, digest):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "summary.json").write_text(json.dumps({
            "actual_epochs": 100, "best_val_frame_map": 0.3, "best_epoch": 80}))
        config = {"optimization": {"epochs": 100, "seed": 42},
                  "train_subset": {"fraction": fraction, "sample_ids_sha256": digest}}
        (directory / "config.json").write_text(json.dumps(config))
        return probing_report.read_run("segmentation", directory, name)
    raw = saved("raw", 0.1, "subset-a")
    matching = saved("layer-3", 0.1, "subset-a")
    fraction = saved("layer-6", 0.01, "subset-a")
    membership = saved("layer-9", 0.1, "subset-b")
    assert probing_report.matched(raw, matching)
    assert not probing_report.matched(raw, fraction)
    assert not probing_report.matched(raw, membership)


@pytest.mark.parametrize("task,budget", [("tmr", 125000), ("segmentation", 10000)])
def test_iteration_report_reads_progress_and_requires_matching_schedule(tmp_path, task, budget):
    def saved(name, warmup=500, final_lr=1e-6):
        directory = tmp_path / name
        directory.mkdir()
        config = {"optimization": {"iterations": budget, "warmup_iterations": warmup,
                                  "eval_interval": 100, "batch_size": 128,
                                  "seed": 42, "start_lr": 0., "lr": 1e-4, "final_lr": final_lr}}
        (directory / "config.json").write_text(json.dumps(config))
        summary = {"global_step": budget, "best_step": 100}
        if task == "segmentation":
            summary["best_val_frame_map"] = .3
            history = [{"iteration": step, "lr": 1e-4,
                        "train": {"loss": .5}, "val": {"loss": .6, "babel-120": {"frame_map": value}}}
                       for step, value in ((100, .3), (budget, .2))]
            (directory / "metrics.json").write_text(json.dumps(history))
        else:
            summary["best_val"] = {"mean_r1": .3}
            (directory / "metrics.csv").write_text(
                f"iteration,train_loss,val_mean_r1\n100,0.5,0.3\n{budget},0.4,0.2\n")
        (directory / "summary.json").write_text(json.dumps(summary))
        return probing_report.read_run(task, directory, name)
    raw = saved("raw")
    assert raw["unit"] == "iteration" and raw["progress"] == budget
    assert raw["best_position"] == 100 and raw["epochs"] is None
    assert raw["complete"]
    assert [point["step"] for point in raw["points"]] == [100, budget]
    assert probing_report.matched(raw, saved("layer-3"))
    assert not probing_report.matched(raw, saved("layer-6", warmup=100))
    assert not probing_report.matched(raw, saved("layer-9", final_lr=1e-5))


def test_suite_forwards_explicit_iteration_recipe(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fixture")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(probe_suite.sys, "argv", ["probe_suite", "--checkpoint", str(checkpoint),
        "--task", "tmr", "--runs", "raw", "--iterations", "7", "--warmup-iterations", "2",
        "--eval-interval", "3", "--checkpoint-interval", "2", "--log-freq", "1", "--batch-size", "4"])
    def run(command, **kwargs):
        if command[2] == "experiment.tmr":
            for key, value in (("iterations", "7"), ("warmup-iterations", "2"),
                               ("eval-interval", "3"), ("checkpoint-interval", "2"),
                               ("log-freq", "1"), ("batch-size", "4")):
                assert command[command.index("--" + key) + 1] == value
            assert "--epochs" not in command and "--warmup-epochs" not in command
            output = Path(command[command.index("--output-root") + 1])
            output.mkdir(parents=True)
            (output / "summary.json").write_text("{}")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(probe_suite.subprocess, "run", run)
    probe_suite.main()
