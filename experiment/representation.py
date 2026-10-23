"""Downstream representation selection from preprocessed dataset metadata."""
import json
from pathlib import Path

from motion_rep.registry import REPRESENTATIONS, get_representation


def add_argument(parser):
    parser.add_argument("--representation", choices=list(REPRESENTATIONS), default=None,
                        help="Motion representation; local selects local default dataset/cache paths. Custom datasets are validated; omitted infers from metadata.")


def validate_dataset(root, representation=None):
    metadata = json.loads((Path(root) / "meta.json").read_text())
    actual = get_representation(metadata["representation"])
    if metadata["motion_dim"] != actual.FEATURE_DIM:
        raise ValueError("Dataset representation and motion_dim disagree")
    if representation is not None and get_representation(representation) is not actual:
        raise ValueError("Requested representation differs from the dataset representation")
    return actual

def select_defaults(args, task):
    """Select the prepared local dataset only when the standard paths are used."""
    if getattr(args, "representation", None) != "LocalMotionJEPARep":
        return
    from experiment.utils import PROJECT_ROOT
    if task == "tmr":
        from experiment.tmr.dataset import DEFAULT_DATASET_ROOT, DEFAULT_CACHE_ROOT
        if args.dataset_root == DEFAULT_DATASET_ROOT:
            args.dataset_root = PROJECT_ROOT / "dataset/bones-seed-local-from366-nframes150-min60"
        if args.cache_root == DEFAULT_CACHE_ROOT:
            args.cache_root = PROJECT_ROOT / "output/tmr-cache-llm2vec-local"
    else:
        if args.dataset_root == PROJECT_ROOT / "dataset/babel-segmentation-120-processed-nframes30-150":
            args.dataset_root = PROJECT_ROOT / "dataset/babel-segmentation-120-local-from366-nframes30-150"
