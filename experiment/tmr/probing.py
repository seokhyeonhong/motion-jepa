"""RAM-only motion feature preparation for attentive retrieval probes."""

from __future__ import annotations

import gc
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from experiment.probing import FeatureMoments, FrozenFrameExtractor


def selected_feature_layers(args):
    layers = getattr(args, "feature_layers", None)
    single = getattr(args, "feature_layer", None)
    if layers is not None and single is not None:
        raise ValueError("Use either --feature-layers or --feature-layer")
    layers = list(layers) if isinstance(layers, (list, tuple)) else ([12 if single is None else single] if layers is None else [])
    if not layers or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in layers):
        raise ValueError("Feature layers must be positive one-based integers")
    if len(set(layers)) != len(layers):
        raise ValueError("Feature layers must not contain duplicates")
    return layers


class FrozenLayersExtractor(FrozenFrameExtractor):
    """Concatenate ordered block outputs, retaining TMR complete-patch masking."""

    def __init__(self, checkpoint, layers, **kwargs):
        from types import SimpleNamespace
        self.layers = selected_feature_layers(SimpleNamespace(feature_layers=layers))
        super().__init__(checkpoint, max(self.layers), **kwargs)
        width = self.feature_dim
        self.feature_dim *= len(self.layers)
        self.metadata.update(feature_layer=None, feature_layers=self.layers,
                             feature_dim=self.feature_dim, layer_feature_dim=width,
                             fusion="channel_concatenation_in_requested_order")

    @torch.inference_mode()
    def __call__(self, motion, fps, lengths):
        if self.encoder is None:
            raise RuntimeError("Frozen extractor was released after preprocessing")
        outputs, handles = {}, []
        def capture(layer):
            def hook(module, inputs, output):
                outputs[layer] = output
            return hook
        try:
            for layer in self.layers:
                handles.append(self.encoder.blocks[layer - 1].register_forward_hook(capture(layer)))
            super().__call__(motion, fps, lengths)
            return torch.cat([outputs[layer] for layer in self.layers], dim=-1).float()
        finally:
            for handle in handles:
                handle.remove()


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
    layers = selected_feature_layers(args) if args.feature_source == "jepa" else None
    result = {
        "feature_source": args.feature_source,
        "feature_layer": layers[0] if layers is not None and len(layers) == 1 else None,
        "checkpoint": str(Path(checkpoint).expanduser().resolve()) if checkpoint is not None else None,
        "encoder_key": args.encoder_key if args.feature_source == "jepa" else None,
        "normalization": "valid_training_frames_channel_mean_std",
        "cache": "anonymous_cpu_float32",
    }

    if layers is not None and len(layers) > 1:
        result["feature_layers"] = layers
    return result

def prepare_probe_datasets(datasets, provenance, configuration, device, *, batch_size=256, num_workers=0):
    """Extract only paired motion clips, fit train-only moments, then release JEPA."""
    source = configuration["feature_source"]
    extractor = None
    raw_dim = int(provenance["motion_dim"])
    try:
        if source == "jepa":
            layers = configuration.get("feature_layers")
            extractor_class = FrozenLayersExtractor if layers is not None else FrozenFrameExtractor
            extractor = extractor_class(
                configuration["checkpoint"], layers if layers is not None else configuration["feature_layer"],
                encoder_key=configuration["encoder_key"], device=device,
                representation=getattr(datasets["train"].raw_dataset, "representation", None),
            )
            feature_dim = extractor.feature_dim
            feature_metadata = dict(extractor.metadata)
        elif source == "raw":
            feature_dim = raw_dim
            feature_metadata = {"feature_source": "raw", "feature_dim": raw_dim}
        else:
            raise ValueError("Unknown probe feature source")
        layout = getattr(extractor, "token_layout", None)
        def token_lengths(lengths):
            lengths = torch.as_tensor(lengths)
            return layout.valid_token_lengths(lengths) if layout is not None else lengths
        moments = FeatureMoments(feature_dim)
        for split, dataset in datasets.items():
            lengths = token_lengths([record["length"] for record in dataset.records]).tolist()
            if any(length < 1 for length in lengths):
                raise ValueError("Every clip must contain a complete motion token")
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
                batch_lengths = token_lengths(batch_lengths)
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
    if layout is not None and layout.patchified:
        result["motion_num_tokens"] = layout.token_num_frames
    result["probe"] = {**configuration, "extractor": feature_metadata}
    print(
        f"RAM motion features ready: {sum(dataset.motion_bank.values.numel() * 4 for dataset in datasets.values()) / 2**30:.2f} GiB; "
        "frozen encoder released before probe construction", flush=True,
    )
    return datasets, result, mean, std
