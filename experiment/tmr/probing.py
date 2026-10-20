"""RAM-only motion feature preparation for attentive retrieval probes."""

from __future__ import annotations

import gc
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from experiment.probing import FeatureMoments, FrozenFrameExtractor


class MemoryTokenBank:
    """Anonymous CPU tensors; no memmap, cache directory, or serialized features."""

    def __init__(self, keys, lengths, feature_dim):
        self.key_to_row = {key: row for row, key in enumerate(keys)}
        if len(self.key_to_row) != len(keys):
            raise ValueError("RAM token keys must be unique")
        self.offsets = torch.cat((torch.zeros(1, dtype=torch.int64), torch.tensor(lengths, dtype=torch.int64).cumsum(0)))
        self.values = torch.empty((int(self.offsets[-1]), feature_dim), dtype=torch.float32)

    def __getitem__(self, key):
        row = self.key_to_row[key]
        return self.values[int(self.offsets[row]):int(self.offsets[row + 1])]


class MotionRows(Dataset):
    def __init__(self, dataset):
        self.base = dataset.raw_dataset
        self.base.normalize = False
        self.rows = [dataset.raw_rows[record["sample_id"]] for record in dataset.records]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.base[self.rows[index]]


def probe_configuration(args):
    if getattr(args, "mode", "supervised") != "probe":
        return None
    checkpoint = getattr(args, "checkpoint", None)
    return {
        "feature_source": args.feature_source,
        "feature_layer": int(args.feature_layer) if args.feature_source == "jepa" else None,
        "checkpoint": str(Path(checkpoint).expanduser().resolve()) if checkpoint is not None else None,
        "encoder_key": args.encoder_key if args.feature_source == "jepa" else None,
        "normalization": "valid_training_frames_channel_mean_std",
        "cache": "anonymous_cpu_float32",
    }


def prepare_probe_datasets(datasets, provenance, configuration, device, *, batch_size=256, num_workers=0):
    """Extract only paired motion clips, fit train-only moments, then release JEPA."""
    source = configuration["feature_source"]
    extractor = None
    raw_dim = int(provenance["motion_dim"])
    try:
        if source == "jepa":
            extractor = FrozenFrameExtractor(
                configuration["checkpoint"], configuration["feature_layer"],
                encoder_key=configuration["encoder_key"], device=device,
            )
            feature_dim = extractor.feature_dim
            feature_metadata = dict(extractor.metadata)
        elif source == "raw":
            feature_dim = raw_dim
            feature_metadata = {"feature_source": "raw", "feature_dim": raw_dim}
        else:
            raise ValueError("Unknown probe feature source")
        moments = FeatureMoments(feature_dim)
        for split, dataset in datasets.items():
            lengths = [record["length"] for record in dataset.records]
            bank = MemoryTokenBank(dataset.sample_ids, lengths, feature_dim)
            loader = DataLoader(
                MotionRows(dataset), batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=device.type == "cuda",
            )
            cursor = 0
            for motion, fps, batch_lengths in tqdm(loader, desc=f"RAM {source} {split}"):
                if extractor is None:
                    features = motion.float()
                else:
                    features = extractor(motion, fps, batch_lengths).detach().float().cpu()
                valid = torch.arange(features.shape[1])[None, :] < batch_lengths[:, None]
                if split == "train":
                    moments.update(features, valid)
                for row, length in enumerate(batch_lengths.tolist()):
                    bank[dataset.sample_ids[cursor]].copy_(features[row, :length])
                    cursor += 1
            if cursor != len(dataset):
                raise RuntimeError("RAM feature extraction missed motion clips")
            dataset.motion_bank = bank
            dataset.raw_dataset = None
            dataset.raw_rows = None
            # The worker loader must not retain its raw dataset or frozen outputs.
            del loader
        mean, std = moments.finalize()
    finally:
        if extractor is not None:
            extractor.release()
        del extractor
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = dict(provenance)
    result["motion_dim"] = feature_dim
    result["probe"] = {**configuration, "extractor": feature_metadata}
    print(
        f"RAM motion features ready: {sum(dataset.motion_bank.values.numel() * 4 for dataset in datasets.values()) / 2**30:.2f} GiB; "
        "frozen encoder released before probe construction", flush=True,
    )
    return datasets, result, mean, std
