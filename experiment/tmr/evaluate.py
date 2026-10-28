"""Evaluate a trained alignment checkpoint against a complete prepared gallery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiment.utils import _atomic_json_save, resolve_device
from .features import load_prepared_datasets
from .model import AlignmentConfig, TextMotionAlignment
from .train import DATA_FIELDS, evaluate_split, load_alignment_checkpoint


def run(args: argparse.Namespace) -> dict[str, float]:
    if args.batch_size < 1 or args.chunk_size < 1 or args.num_workers < 0:
        raise ValueError("Evaluation sizes must be positive and workers nonnegative")
    checkpoint = load_alignment_checkpoint(Path(args.checkpoint).expanduser().resolve())
    config = checkpoint["config"]
    data: dict[str, Any] = dict(config["data"])
    for field in DATA_FIELDS:
        value = getattr(args, field, None)
        if value is not None:
            data[field] = str(value) if isinstance(value, Path) else value
    datasets, provenance = load_prepared_datasets(**data)
    device = resolve_device(args.device)
    if "probe" in config:
        from .probing import prepare_probe_datasets
        datasets, provenance, _, _ = prepare_probe_datasets(
            datasets, provenance, config["probe"], device,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        args.num_workers = 0
    model = TextMotionAlignment(AlignmentConfig(**config["model"])).to(device)
    if "lora" in config:
        from .lora import attach_lora
        provenance = attach_lora(model, datasets, provenance, config["lora"], device)
    if "finetune" in config:
        from .finetune import attach_finetune
        provenance = attach_finetune(model, datasets, provenance, config["finetune"], device)
    if provenance != checkpoint["provenance"]:
        raise ValueError("Evaluation feature provenance differs from the training checkpoint")
    model.load_state_dict(checkpoint["model"], strict=True)
    metrics = evaluate_split(
        model, datasets[args.split], device, batch_size=args.batch_size,
        num_workers=args.num_workers, chunk_size=args.chunk_size,
        use_bfloat16=config["optimization"]["use_bfloat16"],
    )
    if args.output is not None:
        path = Path(args.output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json_save({"checkpoint": str(args.checkpoint), "split": args.split, **metrics}, path)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--annotations-path", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--stats-path", type=Path)
    parser.add_argument(
        "--text-model",
        help="LLM2Vec Llama-3 MNTP model ID/local directory. "
             "Uses the activated environment's Hugging Face cache.",
    )
    parser.add_argument("--text-revision")
    parser.add_argument("--text-adapter", help="LLM2Vec supervised adapter; defaults to the Kimodo Llama-3 adapter.")
    parser.add_argument("--text-adapter-revision")
    parser.add_argument("--max-text-length", type=int)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
