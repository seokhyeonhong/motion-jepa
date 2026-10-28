# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from nv-tlabs/kimodo e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954.
# Batched-only evaluation adapters; see NOTICE.md for changes.
from __future__ import annotations
import torch
import einops

def compute_vel_xyz(
    positions: torch.Tensor,
    fps: float,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the velocities from positions: dx/dt.

    Works with batches. The last velocity is duplicated to keep the same size.
        Args:
            positions (torch.Tensor): [..., T, J, 3] xyz positions of a human skeleton
            fps (float): frame per seconds
            lengths (Optional[torch.Tensor]): [...] size of each input batched. If not provided, positions should not be batched

        Returns:
            velocity (torch.Tensor): [..., T, J, 3] velocities computed from the positions
    """
    device = positions.device

    if lengths is None:
        assert positions.shape[0] == 1, "If lengths is not provided, the input should not be batched."
        lengths = torch.tensor([len(positions)], device=device)

    # compute velocities with fps
    velocity = fps * (positions[:, 1:] - positions[:, :-1])
    # pading the velocity vector
    vel_pad = torch.zeros_like(velocity[:, 0])
    velocity, _ = einops.pack([velocity, vel_pad], "batch * nbjoints dim")

    # repeat the previous velocity at the last valid frame of each sequence. Static-shape
    # formulation: an index_put driven by `lengths` is data-dependent and hostile to
    # torch.compile (this runs inside the compiled denoiser via global_root_to_local_root).
    time_idx = torch.arange(velocity.shape[1], device=device)
    is_last_valid = time_idx[None, :] == (lengths - 1)[:, None]  # [B, T]
    velocity_prev = torch.cat([torch.zeros_like(velocity[:, :1]), velocity[:, :-1]], dim=1)
    velocity = torch.where(is_last_valid[:, :, None, None], velocity_prev, velocity)
    return velocity

def compute_heading_angle(posed_joints: torch.Tensor, skeleton: SkeletonBase) -> torch.Tensor:
    """Compute the heading direction from joint positions using the hip vector.

    Args:
        posed_joints: [B, T, J, 3] global joint positions.
        skeleton: Skeleton instance used to get hip joint indices.

    Returns:
        [B] heading angle in radians.
    """
    # compute root heading for the sequence from hip positions
    r_hip, l_hip = skeleton.hip_joint_idx
    diff = posed_joints[:, :, r_hip] - posed_joints[:, :, l_hip]
    heading_angle = torch.atan2(diff[..., 2], -diff[..., 0])
    return heading_angle

class RotateFeatures:
    """Helper that applies a global heading rotation to motion features."""

    def __init__(self, angle: torch.Tensor):
        """Precompute 2D and 3D rotation matrices for a batch of angles.

        Args:
            angle: Rotation angle(s) in radians, shaped ``[B]``.
        """
        self.angle = angle

        ## Create the necessary rotations matrices
        cos, sin = torch.cos(angle), torch.sin(angle)
        one, zero = torch.ones_like(angle), torch.zeros_like(angle)

        # 2D rotation transposed (sin are -sin)
        self.corrective_mat_2d_T = torch.stack((cos, sin, -sin, cos), -1).reshape(angle.shape + (2, 2))
        # 3D rotation on Y axis
        self.corrective_mat_Y = torch.stack((cos, zero, sin, zero, one, zero, -sin, zero, cos), -1).reshape(
            angle.shape + (3, 3)
        )
        self.corrective_mat_Y_T = self.corrective_mat_Y.transpose(-2, -1).contiguous()

    @staticmethod
    def _align_mat(mat: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        """Insert singleton dims so the per-angle matrices broadcast against ``tensor``.

        ``mat`` has the angle's batch shape plus the trailing matrix dims; ``tensor`` is
        ``[*angle_shape, ..., C]``. A per-sample angle ``[B]`` applied to ``[B, T, J, 3]``
        needs ``[B, 1, 3, 3]`` (plain ``@`` would misalign ``T`` with ``B``); a per-frame
        angle ``[B, T]`` on the same tensor already aligns and is left unchanged.
        """
        pad = tensor.dim() - mat.dim()
        assert pad >= 0, f"angle batch shape {mat.shape[:-2]} has more dims than positions {tensor.shape}"
        return mat.reshape(mat.shape[:-2] + (1,) * pad + mat.shape[-2:])

    def rotate_positions(self, positions: torch.Tensor):
        """Rotate 3D positions around the Y axis."""
        return positions @ self._align_mat(self.corrective_mat_Y_T, positions)

    def rotate_2d_positions(self, positions_2d: torch.Tensor):
        """Rotate 2D ``(x, z)`` vectors in the ground plane."""
        return positions_2d @ self._align_mat(self.corrective_mat_2d_T, positions_2d)

    def rotate_rotations(self, rotations: torch.Tensor):
        """Left-multiply global rotation matrices by the heading correction."""
        # "Rotate" the global rotations
        # which means add an extra Y rotation after the transform
        # so at the left R' = R_y R
        # (since we use the convention x' = R x)
        # "bik,btdkj->btdij"

        B, T, J = rotations.shape[:3]
        BTJ = B * T * J
        return (
            self.corrective_mat_Y[:, None, None].expand(B, T, J, 3, 3).reshape(BTJ, 3, 3) @ rotations.reshape(BTJ, 3, 3)
        ).reshape(B, T, J, 3, 3)



def foot_detect_from_pos_and_vel(
    positions: torch.Tensor,
    velocity: torch.Tensor,
    skeleton,
    vel_thres: float,
    height_thresh: float,
) -> torch.Tensor:
    """Compute foot contact labels using heuristics combining joint height and velocities.

    Args:
        positions (torch.Tensor): [X, T, J, 3] global joint positions
        velocity (torch.Tensor): [X, T, J, 3] velocities (already padded correctly), already multiplied by 1 / dt
        vel_thres (float): threshold for joint velocity
        height_thresh (float): threshold for joint height

    Returns:
        torch.Tensor: [X, T, 4] contact labels for left and right foot joints
        (heel/toe order follows the skeleton joint index definition), where
        ``1`` denotes contact.
    """
    device = positions.device
    # Use at most 2 foot joints per side (ankle + toe); SOMA77 defines a
    # third end-effector (ToeEnd) that SOMA30 and other skeletons omit.
    fid_l = skeleton.left_foot_joint_idx[:2]
    fid_r = skeleton.right_foot_joint_idx[:2]

    velfactor, heightfactor = (
        torch.tensor([vel_thres, vel_thres], device=device),
        torch.tensor([height_thresh, height_thresh], device=device),
    )

    feet_l_v = torch.linalg.norm(velocity[:, :, fid_l], axis=-1)
    feet_l_h = positions[:, :, fid_l, 1]

    feet_l = torch.logical_and(
        feet_l_v < velfactor,
        feet_l_h < heightfactor,
    ).to(positions.dtype)

    feet_r_v = torch.linalg.norm(velocity[:, :, fid_r], axis=-1)
    feet_r_h = positions[:, :, fid_r, 1]

    feet_r = torch.logical_and(
        feet_r_v < velfactor,
        feet_r_h < heightfactor,
    ).to(positions.dtype)

    foot_contacts = torch.cat((feet_l, feet_r), axis=-1)
    return foot_contacts
