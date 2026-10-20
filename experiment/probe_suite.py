"""Run the requested attentive-probe comparisons sequentially without feature files."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", default="probing-base-allctx-200000-20261018")
    parser.add_argument("--task", choices=("all", "segmentation", "tmr"), default="all")
    parser.add_argument("--runs", nargs="+", default=["raw", "3", "6", "9", "12"])
    parser.add_argument("--report-start", type=int, default=1, help="First sequential findings number; TMR uses the following number")
    parser.add_argument("--report-suffix", default="")
    parser.add_argument("--comparison-name", help="Prior probe suite to compare against in the reports")
    args = parser.parse_args()
    root = Path.cwd()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    unknown = set(args.runs) - {"raw", "3", "6", "9", "12"}
    if unknown:
        raise ValueError(f"Unknown comparisons: {unknown}")
    tasks = ("segmentation", "tmr") if args.task == "all" else (args.task,)
    for task in tasks:
        task_root = root / "output" / task / args.name
        report_number = args.report_start + int(task == "tmr")
        report_dir = root / "findings" / f"{report_number:03d}-{task}-probing{args.report_suffix}"
        for selected in args.runs:
            name = "raw" if selected == "raw" else "layer-" + selected
            output = task_root / name / "seed-42"
            summary = output / "summary.json"
            if summary.is_file():
                print(f"Already complete: {task}/{name}", flush=True)
                continue
            command = [sys.executable, "-m", "experiment." + task,
                "--mode", "probe", "--feature-source", "raw" if selected == "raw" else "jepa",
                "--model-size", "base", "--epochs", "100", "--warmup-epochs", "5",
                "--batch-size", "256", "--seed", "42", "--num-workers", "8",
                "--precompute-batch-size", "256", "--probe-heads", "6",
                "--weight-decay", "0.01", "--gradient-clip", "1.0",
                "--lr", "0.0003" if task == "segmentation" else "0.0001",
                "--output-root", str(output), "--device", "cuda:0"]
            if selected != "raw":
                command += ["--checkpoint", str(checkpoint), "--feature-layer", selected,
                            "--encoder-key", "target_encoder"]
            if task == "tmr":
                command += ["--eval-batch-size", "256", "--final-lr", "0.000001",
                            "--temperature", "0.1", "--use-bfloat16", "--tensorboard"]
            if (output / "latest.pth.tar").is_file():
                command += ["--resume"]
            # Keep process status separate from trainer output existence checks.
            status_path = task_root / (name + "-status.json")
            status = {"task": task, "run": name, "state": "running", "started_at": stamp(),
                      "output": str(output), "command": command}
            write_json(status_path, status)
            print(f"Starting {task}/{name}: {stamp()}", flush=True)
            env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
            completed = subprocess.run(command, env=env)
            status.update(ended_at=stamp(), return_code=completed.returncode,
                          state="completed" if completed.returncode == 0 and summary.is_file() else "failed")
            write_json(status_path, status)
            if status["state"] != "completed":
                raise RuntimeError(f"{task}/{name} did not complete; preserving checkpoints and stopping queue")
            comparison_args = (["--comparison-root", str(root / "output" / task / args.comparison_name)]
                               if args.comparison_name else [])
            subprocess.run([sys.executable, "-m", "experiment.probing_report", "--task", task,
                            "--run-root", str(task_root), "--report-dir", str(report_dir),
                            "--checkpoint", str(checkpoint), *comparison_args], check=True, env=env)
        print(f"Finished all requested {task} comparisons: {stamp()}", flush=True)


if __name__ == "__main__":
    main()
