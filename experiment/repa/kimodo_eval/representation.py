# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from nv-tlabs/kimodo e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954.
# Batched-only evaluation adapters; see NOTICE.md for changes.
import torch
import numpy as np
import einops
from .features import RotateFeatures, compute_heading_angle, compute_vel_xyz, foot_detect_from_pos_and_vel


class TMRMotionRep(torch.nn.Module):
    """Motion-only adapter for Kimodo's SOMA30 TMR feature representation."""
    motion_rep_dim = 186

    def __init__(self, skeleton, stats_path, fps=30):
        super().__init__()
        self.skeleton, self.fps = skeleton, fps
        self.ps = [torch.Size([3]), torch.Size([2]), torch.Size([29,3]),
                   torch.Size([30,3]), torch.Size([4])]
        self.slice_dict = {"root_pos": slice(0,3)}
        mean = np.load(stats_path / "mean.npy", allow_pickle=False)
        std = np.load(stats_path / "std.npy", allow_pickle=False)
        if mean.shape != (186,) or std.shape != (186,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std < 0).any():
            raise ValueError("Invalid Kimodo TMR normalization statistics")
        self.register_buffer("mean", torch.as_tensor(mean).float(), persistent=False)
        self.register_buffer("std_eps", torch.sqrt(torch.as_tensor(std).float().square()+1e-5), persistent=False)

    def normalize(self, features):
        return (features-self.mean)/self.std_eps

    def canonicalize(self, features, normalized=False):
        if normalized:
            raise ValueError("Canonicalize raw features before normalization")
        angle = torch.atan2(features[:,0,4], features[:,0,3])
        rotated = self.rotate(features, -angle)
        return self.translate_2d(rotated, -rotated[:,0,[0,2]])

    def __call__(
        self,
        local_joint_rots: torch.Tensor | None = None,
        root_positions: torch.Tensor | None = None,
        posed_joints: torch.Tensor | None = None,
        *,
        to_normalize: bool,
        to_canonicalize: bool = False,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convert motion inputs to this feature representation.

        Args:
            local_joint_rots: Local joint rotation matrices ``[B, T, J, 3, 3]``.
                Required when ``posed_joints`` is not provided.
            root_positions: Root translations ``[B, T, 3]``. Required when
                ``posed_joints`` is not provided.
            posed_joints: Optional precomputed global joint positions
                ``[B, T, J, 3]``. If passed, FK is skipped.
            to_normalize: Whether to normalize output features.
            to_canonicalize: Whether to canonicalize output features (False by default).
            lengths: Optional valid lengths for variable-length batches.

        Returns:
            Motion features with shape ``[B, T, motion_rep_dim]``.
        """
        if posed_joints is not None:
            device = posed_joints.device
            nbatch, nbframes, nbjoints = posed_joints.shape[:3]
        else:
            device = local_joint_rots.device
            nbatch, nbframes, nbjoints = local_joint_rots.shape[:3]

        if lengths is None:
            assert nbatch == 1, "If lenghts is not provided, the input should not be batched."
            lengths = torch.tensor([nbframes], device=device)

        if posed_joints is None:
            raise ValueError("Evaluation requires decoded posed_joints")
        else:
            global_positions = posed_joints
            root_positions = posed_joints[:, :, 0]
            local_joints_positions_origin_is_pelvis = posed_joints - root_positions[:, :, None]

        root_heading_angle = compute_heading_angle(global_positions, self.skeleton)
        global_root_heading = torch.stack([torch.cos(root_heading_angle), torch.sin(root_heading_angle)], dim=-1)

        ground_offset = 0 * root_positions
        ground_offset[..., 1] = root_positions[..., 1]

        local_joints_positions = local_joints_positions_origin_is_pelvis[:, :, 1:] + ground_offset[:, :, None]
        velocities = compute_vel_xyz(global_positions, self.fps, lengths=lengths)

        # Remove the heading angle for each frame
        RF = RotateFeatures(-root_heading_angle)
        local_joints_positions = RF.rotate_positions(local_joints_positions)
        velocities = RF.rotate_positions(velocities)

        foot_contacts = foot_detect_from_pos_and_vel(global_positions, velocities, self.skeleton, 0.15, 0.10)
        features, _ = einops.pack(
            [
                root_positions,
                global_root_heading,
                local_joints_positions,
                velocities,
                foot_contacts,
            ],
            "batch time *",
        )

        if to_canonicalize:
            features = self.canonicalize(features, normalized=False)

        if to_normalize:
            features = self.normalize(features)
        return features

    def rotate(self, features: torch.Tensor, angle: torch.Tensor):
        """Rotate all spatial features by a heading delta (radians)."""
        # rotate by the angle
        # it add the angle to the current features
        # assume it is not normalized
        bs = features.shape[0]
        device = features.device
        [
            root_pos,
            global_root_heading,
            local_joints_positions,
            velocities,
            foot_contacts,
        ] = einops.unpack(features, self.ps, "batch time *")

        if not isinstance(angle, torch.Tensor):
            angle = torch.tensor(angle, device=device)
        if len(angle.shape) == 0:
            angle = angle.repeat(bs)

        RF = RotateFeatures(angle)
        new_features, _ = einops.pack(
            [
                RF.rotate_positions(root_pos),
                RF.rotate_2d_positions(global_root_heading),
                local_joints_positions,  # already rotation invariant
                velocities,  # already rotation invariant
                foot_contacts,
            ],
            "batch time *",
        )
        return new_features

    def translate_2d(
        self,
        features: torch.Tensor,
        translation_2d: torch.Tensor,
    ) -> torch.Tensor:
        """Translate root planar position by ``(dx, dz)``."""
        # only move on the ground
        # For 3D, we should not forget to move the local_joints_positions as well
        bs = features.shape[0]
        if len(translation_2d.shape) == 1:
            translation_2d = translation_2d.repeat(bs, 1)

        new_features = features.clone()
        new_root_pos = new_features[:, :, self.slice_dict["root_pos"]]
        new_root_pos[:, :, 0] += translation_2d[:, [0]]
        new_root_pos[:, :, 2] += translation_2d[:, [1]]
        return new_features
