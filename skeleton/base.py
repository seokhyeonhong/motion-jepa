"""Base skeleton class: hierarchy, assets, and kinematic helpers."""

from pathlib import Path

import numpy as np
import torch

from .asset_paths import SKELETON_METADATA_PATH, skeleton_asset_path
from .kinematics import fk
from .transforms import to_standard_tpose


class SkeletonBase(torch.nn.Module):
    """Common skeleton metadata and asset loading."""

    name: str
    metadata_key: str

    def __init__(self, folder: str | Path | None = None, load: bool = True, **kwargs):
        super().__init__()
        del kwargs
        folder = Path(folder) if folder is not None else skeleton_asset_path(self.name)
        self.folder = str(folder)
        with np.load(SKELETON_METADATA_PATH, allow_pickle=False) as data:
            self.names = list(data[f"{self.metadata_key}_names"].astype(str))
            parents = torch.from_numpy(
                data[f"{self.metadata_key}_parents"].astype(np.int64)
            )
            subset_indices = torch.from_numpy(data["soma30_indices"].astype(np.int64))
        self.register_buffer("parents", parents, persistent=False)
        self.register_buffer("subset_indices", subset_indices, persistent=False)
        self.nbjoints = len(self.names)
        self.dim = self.nbjoints
        self.root_idx = 0
        self.name_to_index = {name: index for index, name in enumerate(self.names)}
        self.bone_index = self.name_to_index
        self.left_foot_joint_idx = [
            self.name_to_index["LeftFoot"],
            self.name_to_index["LeftToeBase"],
        ]
        self.right_foot_joint_idx = [
            self.name_to_index["RightFoot"],
            self.name_to_index["RightToeBase"],
        ]
        self.foot_joint_idx = self.left_foot_joint_idx + self.right_foot_joint_idx
        self.hip_joint_idx = [
            self.name_to_index["RightLeg"],
            self.name_to_index["LeftLeg"],
        ]
        if not load:
            return
        self.register_buffer(
            "neutral_joints",
            torch.load(folder / "joints.p", map_location="cpu", weights_only=True).squeeze(),
            persistent=False,
        )
        offsets_path = folder / "standard_t_pose_global_offsets_rots.p"
        if offsets_path.is_file():
            self.register_buffer(
                "global_rot_offsets",
                torch.load(
                    offsets_path, map_location="cpu", weights_only=True
                ).squeeze(),
                persistent=False,
            )
        else:
            self.global_rot_offsets = None
        relaxed_hands_path = folder / "relaxed_hands_rest_pose.npy"
        if relaxed_hands_path.is_file():
            self.register_buffer(
                "relaxed_hands",
                torch.from_numpy(np.load(relaxed_hands_path)).squeeze(),
                persistent=False,
            )
        else:
            self.relaxed_hands = None

    def fk(self, local_rotations, root_positions):
        return fk(local_rotations, root_positions, self.neutral_joints, self.parents)

    def to_standard_tpose(self, local_rotations):
        if self.global_rot_offsets is None:
            raise ValueError(f"{self.name} has no standard T-pose transform.")
        return to_standard_tpose(
            local_rotations, self.parents, self.global_rot_offsets
        )


__all__ = ["SkeletonBase"]
