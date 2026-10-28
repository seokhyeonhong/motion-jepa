"""RAM-only frame features for segmentation attentive probes."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset

from experiment.probing import FeatureMoments, FrozenFrameExtractor


def selected_feature_layers(args):
    """Return the ordered one-based layer list, accepting the legacy scalar."""
    layers = getattr(args, "feature_layers", None)
    single = getattr(args, "feature_layer", None)
    if layers is not None and single is not None:
        raise ValueError("Use either --feature-layers or --feature-layer")
    if layers is None:
        return None if single is None else [single]
    if not isinstance(layers, (list, tuple)) or not layers:
        raise ValueError("feature_layers must be a nonempty list")
    if any(isinstance(layer, bool) or not isinstance(layer, int) or layer <= 0 for layer in layers):
        raise ValueError("feature_layers must contain positive one-based integers")
    if len(set(layers)) != len(layers):
        raise ValueError("feature_layers must not contain duplicates")
    return list(layers)


class FrozenPatchLayersExtractor(FrozenFrameExtractor):
    """Capture selected pre-final-norm block outputs in one frozen forward pass."""

    def __init__(self, checkpoint, layers, **kwargs):
        layers = selected_feature_layers(SimpleNamespace(feature_layers=layers))
        super().__init__(checkpoint, max(layers), **kwargs)
        if not self.token_layout.patchified or self.num_frames % 3:
            self.release()
            raise ValueError("patch-transformer requires a three-frame patch JEPA checkpoint")
        self.layers = layers
        self.layer_feature_dim = self.feature_dim
        self.feature_dim *= len(layers)
        self.metadata.update(
            feature_layers=layers, feature_layer=layers[0] if len(layers) == 1 else None,
            feature_dim=self.feature_dim, layer_feature_dim=self.layer_feature_dim,
            fusion="channel_concatenation_in_requested_order",
            patch_validity="any_valid_frame_zero_padded",
        )

    @torch.inference_mode()
    def __call__(self, motion, fps, lengths):
        if self.encoder is None:
            raise RuntimeError("Frozen extractor was released after preprocessing")
        motion = torch.as_tensor(motion, device=self.device, dtype=torch.float32)
        lengths = torch.as_tensor(lengths, device=self.device)
        if motion.ndim != 3 or motion.shape[1:] != (self.num_frames, self.input_dim):
            raise ValueError("Raw motion shape differs from the checkpoint")
        if lengths.shape != (len(motion),) or ((lengths < 1) | (lengths > self.num_frames)).any():
            raise ValueError("Invalid per-clip frame lengths")
        valid = torch.arange(self.num_frames, device=self.device)[None] < lengths[:, None]
        motion = motion.masked_fill(~valid[..., None], 0)
        if not torch.isfinite(motion).all():
            raise ValueError("Valid raw motion frames must be finite")
        motion = ((motion - self.mean) / self.std).masked_fill(~valid[..., None], 0)
        fps = torch.as_tensor(fps, device=self.device, dtype=torch.float32)
        if fps.shape != (len(motion),) or not torch.isfinite(fps).all() or (fps <= 0).any():
            raise ValueError("Expected positive finite FPS per clip")
        active = valid.reshape(len(motion), -1, 3).any(-1)
        outputs, handles = {}, []
        def capture(layer):
            def hook(module, inputs, output):
                outputs[layer] = output
            return hook
        try:
            for layer in self.layers:
                handles.append(self.encoder.blocks[layer - 1].register_forward_hook(capture(layer)))
            self.encoder(motion, fps, valid_frames=active.repeat_interleave(3, dim=1),
                         feature_layer=max(self.layers))
            return torch.cat([outputs[layer] for layer in self.layers], dim=-1).float().masked_fill(
                ~active[..., None], 0)
        finally:
            for handle in handles:
                handle.remove()


class SegmentationFeatureDataset(Dataset):
    """Keep features and supervision in CPU memory throughout probe training."""

    def __init__(self, source, feature_dim, patch=False):
        self.num_classes = source.num_classes
        self.class_indices_60 = source.class_indices_60
        self.num_frames = source.num_frames
        self.features = torch.empty(len(source), source.num_frames // 3 if patch else source.num_frames, feature_dim)
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
    layers = selected_feature_layers(args)
    patch = getattr(args, "probe_architecture", "attentive") in {"patch-transformer", "mlp"}
    if getattr(args, "feature_layers", None) is not None and not patch:
        raise ValueError("--feature-layers requires --probe-architecture patch-transformer")
    if patch and args.feature_source != "jepa":
        raise ValueError("patch-transformer requires JEPA features")
    extractor = None
    if args.feature_source == "jepa":
        if args.checkpoint is None or layers is None:
            raise ValueError("JEPA probing requires --checkpoint and --feature-layer or --feature-layers")
        extractor_class = FrozenPatchLayersExtractor if patch else FrozenFrameExtractor
        extractor = extractor_class(
            args.checkpoint, layers if patch else layers[0], encoder_key=args.encoder_key, device=device,
            representation=getattr(datasets["train"], "representation", None),
        )
        feature_dim = extractor.feature_dim
        metadata = dict(extractor.metadata)
        if extractor.num_frames != datasets["train"].num_frames:
            extractor.release()
            raise ValueError("Checkpoint and segmentation frame counts differ")
    else:
        if args.checkpoint is not None or layers is not None:
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
            target = SegmentationFeatureDataset(source, feature_dim, patch=patch)
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
                if patch:
                    valid = valid.reshape(len(motion), -1, 3).any(-1)
                if features.shape != (len(motion), valid.shape[1], feature_dim):
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
    if patch:
        metadata.update(
            layers=layers, feature_frames=datasets["train"].num_frames // 3,
            normalization="population_mean_std_per_layer_channel_over_valid_training_patches",
            training_tokens=moments.count,
        )
    print("Feature preprocessing complete; frozen encoder released before probe training", flush=True)
    return cached, mean, std, metadata
