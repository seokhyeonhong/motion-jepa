"""Prepare reusable frozen text tokens paired with raw motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import DEFAULT_ANNOTATIONS_PATH, DEFAULT_CACHE_ROOT, DEFAULT_DATASET_ROOT, DEFAULT_TEXT_MODEL
from .features import prepare_caches


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    from experiment.representation import add_argument
    add_argument(parser)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--annotations-path", type=Path, default=DEFAULT_ANNOTATIONS_PATH)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--stats-path", type=Path)
    parser.add_argument(
        "--text-model", default=DEFAULT_TEXT_MODEL,
        help="LLM2Vec Llama-3 MNTP model ID/local directory. "
             "Uses the activated environment's Hugging Face cache.",
    )
    parser.add_argument("--text-revision")
    parser.add_argument("--text-adapter", help="LLM2Vec supervised adapter; defaults to the Kimodo Llama-3 adapter.")
    parser.add_argument("--text-adapter-revision")
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--text-batch-size", type=int, default=8)
    parser.add_argument("--recompute-features", action="store_true")
    parser.add_argument("--max-samples-per-split", type=int)
    return parser


def main(argv=None):
    from experiment.representation import select_defaults
    args = build_parser().parse_args(argv)
    select_defaults(args, "tmr")
    metadata = prepare_caches(**vars(args))
    summary = {key: metadata[key] for key in (
        "input_source", "motion_dim", "text_dim", "split_counts", "filtered_counts"
    )}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
