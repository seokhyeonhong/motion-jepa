"""Frozen JEPA teacher with its original preprocessing."""
import torch
from experiment.probing import FrozenFrameExtractor


class AlignmentTeacher:
    def __init__(self, config, data, mean, std, device):
        if not config.get("checkpoint") or not config.get("config"):
            raise ValueError("Alignment requires explicit teacher checkpoint and config")
        self.extractor = FrozenFrameExtractor(
            config["checkpoint"], config.get("layer"), device=device,
            config_path=config["config"], representation=data["representation"])
        if self.extractor.num_frames != data["num_frames"]:
            raise ValueError("Teacher and generator frame capacities must match")
        self.mean = torch.as_tensor(mean, device=device)
        self.std = torch.as_tensor(std, device=device)
        self.patch_size = self.extractor.token_layout.temporal_patch_size
        self.feature_dim = self.extractor.feature_dim
        self.metadata = self.extractor.metadata

    def __call__(self, clean, fps, valid):
        # Extractor receives raw motion and applies teacher-specific normalization.
        lengths = valid.sum(1)
        prefix = torch.arange(valid.shape[1], device=valid.device)[None] < lengths[:, None]
        if not torch.equal(valid, prefix):
            raise ValueError("Teacher requires contiguous valid frames followed by padding")
        usable = valid.shape[1] // self.patch_size * self.patch_size
        mask = valid[:, :usable].reshape(len(valid), -1, self.patch_size).all(-1)
        if not mask.any(1).all():
            raise ValueError("Each sequence must contain a complete teacher token")
        raw = clean.float() * self.std + self.mean
        result = self.extractor(raw, fps, lengths)
        # Convert inference tensors into ordinary detached tensors for autograd losses.
        return result.clone(), mask
