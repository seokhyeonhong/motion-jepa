"""Run with python -m experiment.repa."""
import argparse
from pathlib import Path
import yaml
from .runner import train, evaluate_checkpoint, sample_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--resume")
    training.add_argument("--max-steps", type=int, help="Bound additional steps without changing the configured schedule")
    training.add_argument("--teacher-checkpoint")
    training.add_argument("--teacher-config")
    training.add_argument("--alignment-weight", type=float)
    training.add_argument("--alignment-loss", choices=["mse", "cosine"])
    training.add_argument("--shortcut-fraction", type=float)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--output", required=True)
    sampling = commands.add_parser("sample")
    sampling.add_argument("--checkpoint", required=True)
    sampling.add_argument("--output", required=True)
    sampling.add_argument("--steps", type=int, default=32)
    sampling.add_argument("--count", type=int, default=16)
    sampling.add_argument("--length", type=int, default=150)
    sampling.add_argument("--fps", type=float, default=30.)
    sampling.add_argument("--seed", type=int, default=0)
    for command in (training, evaluation, sampling):
        command.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.command == "train":
        config = yaml.safe_load(args.config.read_text())
        for attr, key in (("teacher_checkpoint", "checkpoint"), ("teacher_config", "config"),
                          ("alignment_weight", "weight"), ("alignment_loss", "loss")):
            value = getattr(args, attr)
            if value is not None:
                config["alignment"][key] = value
        if args.shortcut_fraction is not None:
            config["shortcut"]["fraction"] = args.shortcut_fraction
        if args.max_steps is not None and args.max_steps < 1:
            parser.error("--max-steps must be positive")
        train(config, args.output, args.resume, args.device, args.max_steps)
    elif args.command == "evaluate":
        evaluate_checkpoint(args.checkpoint, args.output, args.device)
    else:
        sample_checkpoint(args.checkpoint, args.output, args.steps, args.count, args.length, args.fps, args.seed, args.device)


if __name__ == "__main__":
    main()
