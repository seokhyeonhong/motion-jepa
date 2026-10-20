"""RAM-only frame features for segmentation attentive probes."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from experiment.probing import FeatureMoments, FrozenFrameExtractor


class SegmentationFeatureDataset(Dataset):
    """Keep features and supervision in CPU memory for all probe epochs."""

    def __init__(self, source, feature_dim):
        self.num_classes = source.num_classes
        self.class_indices_60 = source.class_indices_60
        self.num_frames = source.num_frames
        self.features = torch.empty(len(source), source.num_frames, feature_dim)
        self.labels = torch.empty(len(source), source.num_frames, source.num_classes, dtype=torch.bool)
        self.supervised = torch.empty(len(source), source.num_frames, dtype=torch.bool)
        self.fps = torch.empty(len(source), dtype=torch.float32)
        self.lengths = torch.empty(len(source), dtype=torch.long)
        self.sample_ids = []

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return (self.features[index], self.fps[index], self.lengths[index],
                self.labels[index], self.supervised[index], self.sample_ids[index])


def prepare_probe_datasets(datasets, args, device):
    """Read each task sample once; fit moments only to valid training frames."""
    if args.precompute_batch_size <= 0 or args.probe_heads <= 0:
        raise ValueError("precompute_batch_size and probe_heads must be positive")
    extractor = None
    if args.feature_source == "jepa":
        if args.checkpoint is None or args.feature_layer is None:
            raise ValueError("JEPA probing requires --checkpoint and --feature-layer")
        extractor = FrozenFrameExtractor(
            args.checkpoint, args.feature_layer, encoder_key=args.encoder_key, device=device,
        )
        feature_dim = extractor.feature_dim
        metadata = dict(extractor.metadata)
        if extractor.num_frames != datasets["train"].num_frames:
            extractor.release()
            raise ValueError("Checkpoint and segmentation frame counts differ")
    else:
        if args.checkpoint is not None or args.feature_layer is not None:
            raise ValueError("Raw probing does not accept a checkpoint or feature layer")
        feature_dim = datasets["train"].motion_dim
        metadata = {}
    if feature_dim % args.probe_heads:
        if extractor is not None:
            extractor.release()
        raise ValueError("Feature dimension must be divisible by probe_heads")
    moments = FeatureMoments(feature_dim)
    cached = {}
    try:
        for split, source in datasets.items():
            target = SegmentationFeatureDataset(source, feature_dim)
            loader = DataLoader(
                source, batch_size=args.precompute_batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=device.type == "cuda",
            )
            offset = 0
            for motion, fps, lengths, labels, supervised, sample_ids in loader:
                with torch.inference_mode():
                    features = extractor(motion, fps, lengths) if extractor is not None else motion
                features = features.detach().float().cpu()
                valid = torch.arange(source.num_frames)[None] < lengths[:, None]
                if features.shape != (*motion.shape[:2], feature_dim):
                    raise ValueError("Feature extraction changed segmentation frame alignment")
                if not torch.isfinite(features[valid]).all():
                    raise ValueError("Non-finite segmentation probe features")
                features = features.masked_fill(~valid.unsqueeze(-1), 0)
                end = offset + len(features)
                target.features[offset:end].copy_(features)
                target.labels[offset:end].copy_(labels.bool())
                target.supervised[offset:end].copy_(supervised)
                target.fps[offset:end].copy_(fps)
                target.lengths[offset:end].copy_(lengths)
                target.sample_ids.extend(sample_ids)
                if split == "train":
                    moments.update(features, valid)
                offset = end
                if offset == len(source) or offset % (10 * args.precompute_batch_size) == 0:
                    print(f"precompute split={split} samples={offset}/{len(source)}", flush=True)
            cached[split] = target
    finally:
        if extractor is not None:
            extractor.release()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    mean, std = moments.finalize()
    metadata.update({
        "source": args.feature_source, "layer": args.feature_layer,
        "feature_dim": feature_dim, "storage": "cpu_ram_only",
        "normalization": "population_mean_std_over_valid_training_frames",
        "training_frames": int(cached["train"].lengths.sum()),
        "split_samples": {split: len(data) for split, data in cached.items()},
    })
    print("Feature preprocessing complete; frozen encoder released before probe training", flush=True)
    return cached, mean, std, metadata
