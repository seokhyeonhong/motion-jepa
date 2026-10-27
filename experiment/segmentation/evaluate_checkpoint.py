"""Re-evaluate a raw supervised checkpoint without changing training history."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset.babel_segmentation import BabelSegmentationDataset, file_sha256
from experiment.utils import _atomic_json_save
from .model import MotionSegmentationTransformer
from .train import evaluate_split


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    signature = checkpoint["signature"]
    arch = signature["architecture"]
    if arch["model_version"] not in {"raw_rope_v1", "raw_rope_patch_v1"}:
        raise ValueError("This evaluator supports raw supervised checkpoints")
    root = args.dataset_root or Path(signature["dataset_root"])
    for name, expected in signature["data_hashes"].items():
        if file_sha256(root / name) != expected:
            raise ValueError(f"Checkpoint dataset hash mismatch: {name}")
    meta = json.loads((root / "meta.json").read_text())
    dataset = BabelSegmentationDataset(
        root, "val", num_frames=arch["num_frames"], fps=meta["fps"],
        motion_dim=arch["input_dim"],
        representation=meta.get("representation", "MotionJEPAMotionRep"))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    model = MotionSegmentationTransformer(
        arch["model_size"], input_dim=arch["input_dim"],
        num_classes=arch["num_classes"], num_frames=arch["num_frames"],
        motion_depth=arch["depth"], patchify=arch.get("patchify", False))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(args.device)
    metrics = evaluate_split(model, loader, torch.device(args.device))
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "step": checkpoint["global_step"], "split": "val",
        "samples": len(dataset), "parameters": model.parameter_counts(),
        "protocol": {
            "prediction_threshold": 0.5,
            "macro_f1": "all vocabulary classes; undefined class F1 = 0",
            "segment_f1": "micro over classes; independent multilabel intervals per window; ignored frames split intervals; chronological max-IoU one-to-one matching",
            "iou_thresholds": [0.1, 0.25, 0.5],
            "precision": "float32",
        },
        "metrics": metrics,
    }
    output = args.output or args.checkpoint.parent / "extended-validation-metrics.json"
    _atomic_json_save(result, output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
