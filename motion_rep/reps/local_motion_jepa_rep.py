"""Heading-local body features with separately preserved root dynamics.

SOMA30 layout (365D): root height, heading-local root XZ velocity, yaw
velocity, pelvis-relative joint positions, heading-relative global 6D
rotations, heading-relative global joint velocities, and foot contacts.
Velocities use forward differences in units/second (radians/second for yaw);
the last valid difference is repeated, matching the existing representation.
Initial planar position and yaw are discarded. Inverse fixes both to zero.
Inputs contain valid frames only; encode padded sequences individually.
"""

from __future__ import annotations

import torch

from skeleton import SOMASkeleton30, global_rots_to_local_rots

from ..geometry import cont6d_to_matrix, matrix_to_cont6d, velocity, y_rotation
from ..feet import foot_detect_from_pos_and_vel
from .base import MotionRepBase


class LocalMotionJEPARep(MotionRepBase):
    FORMAT_NAME = "local_motion_jepa_365_v1"
    FEATURE_DIM = 365

    ROOT_HEIGHT = slice(0, 1)
    ROOT_VELOCITY = slice(1, 3)
    ROOT_YAW_VELOCITY = slice(3, 4)
    LOCAL_POSITIONS = slice(4, 91)
    LOCAL_ROTATIONS = slice(91, 271)
    VELOCITIES = slice(271, 361)
    FOOT_CONTACTS = slice(361, 365)

    def __init__(self, skeleton: SOMASkeleton30 | None = None, fps: int = 30):
        super().__init__(skeleton or SOMASkeleton30(), fps)
        if self.skeleton.nbjoints != 30:
            raise ValueError("LocalMotionJEPARep requires SOMA30.")
        if self.fps <= 0:
            raise ValueError("fps must be positive.")

    def __call__(
        self,
        local_rotations: torch.Tensor,
        root_positions: torch.Tensor,
        to_canonicalize: bool = True,
    ) -> torch.Tensor:
        """Encode motion; heading/translation invariance is intrinsic.

        to_canonicalize is accepted for interface compatibility; either value
        produces the same features because no absolute XZ or yaw is stored.
        """
        unbatched = local_rotations.ndim == 4
        if unbatched:
            local_rotations = local_rotations.unsqueeze(0)
            root_positions = root_positions.unsqueeze(0)
        if local_rotations.shape[1] == 0:
            raise ValueError("Motion must contain at least one frame.")
        # BVH rotations may be float64 while root positions are float32.
        dtype = torch.promote_types(local_rotations.dtype, root_positions.dtype)
        local_rotations = local_rotations.to(dtype=dtype)
        root_positions = root_positions.to(dtype=dtype)
        global_rotations, global_positions, _ = self.skeleton.fk(
            local_rotations, root_positions
        )
        right_hip, left_hip = self.skeleton.hip_joint_idx
        difference = global_positions[:, :, right_hip] - global_positions[:, :, left_hip]
        angle = torch.atan2(difference[..., 2], -difference[..., 0])
        rotation = y_rotation(-angle)
        positions = global_positions[:, :, 1:] - root_positions[:, :, None]
        positions = torch.matmul(rotation[:, :, None], positions.unsqueeze(-1)).squeeze(-1)
        rotations = rotation[:, :, None] @ global_rotations
        global_velocities = velocity(global_positions, self.fps)
        velocities = torch.matmul(
            rotation[:, :, None], global_velocities.unsqueeze(-1)
        ).squeeze(-1)
        root_velocity = torch.matmul(
            rotation, velocity(root_positions, self.fps).unsqueeze(-1)
        ).squeeze(-1)
        if angle.shape[1] > 1:
            delta = angle[:, 1:] - angle[:, :-1]
            yaw_velocity = torch.atan2(torch.sin(delta), torch.cos(delta)) * self.fps
            yaw_velocity = torch.cat([yaw_velocity, yaw_velocity[:, -1:]], dim=1)
        else:
            yaw_velocity = torch.zeros_like(angle)
        contacts = foot_detect_from_pos_and_vel(
            global_positions, global_velocities, self.skeleton, 0.15, 0.10
        )
        features = torch.cat(
            [
                root_positions[..., 1:2],
                root_velocity[..., [0, 2]],
                yaw_velocity.unsqueeze(-1),
                positions.flatten(-2),
                matrix_to_cont6d(rotations).flatten(-2),
                velocities.flatten(-2),
                contacts,
            ],
            dim=-1,
        )
        return features[0] if unbatched else features

    def from_global_features(self, features: torch.Tensor) -> torch.Tensor:
        """Convert unnormalized 366D features without FK or BVH parsing.

        Accepts [T, 366] or [B, T, 366] valid-frame tensors. Existing global
        velocities are rotated, not recomputed from quantized positions.
        """
        from .motion_jepa_motionrep import MotionJEPAMotionRep as Global

        if features.ndim not in (2, 3) or features.shape[-1] != Global.FEATURE_DIM:
            raise ValueError("Expected [T, 366] or [B, T, 366] features.")
        if features.shape[-2] == 0 or not torch.is_floating_point(features):
            raise ValueError("Features must be nonempty floating-point motions.")
        if not torch.isfinite(features).all():
            raise ValueError("Features contain NaN or infinity.")
        unbatched = features.ndim == 2
        if unbatched:
            features = features.unsqueeze(0)
        batch, frames = features.shape[:2]
        heading = features[..., Global.ROOT_HEADING]
        if not torch.allclose(heading.square().sum(-1), torch.ones_like(heading[..., 0]),
                              atol=1e-4, rtol=1e-4):
            raise ValueError("Expected unnormalized unit heading features.")
        angle = torch.atan2(heading[..., 1], heading[..., 0])
        rotation = y_rotation(-angle)
        height = features[..., 1:2]
        positions = features[..., Global.LOCAL_POSITIONS].reshape(batch, frames, 29, 3).clone()
        # Original positions retain height above ground; local positions use pelvis origin.
        positions[..., 1] -= height
        positions = (rotation[:, :, None] @ positions.unsqueeze(-1)).squeeze(-1)
        # 6D rotation encoding stores two columns, so rotate each directly.
        columns = features[..., Global.GLOBAL_ROTATIONS].reshape(batch, frames, 30, 2, 3)
        columns = (rotation[:, :, None, None] @ columns.unsqueeze(-1)).squeeze(-1)
        velocities = features[..., Global.VELOCITIES].reshape(batch, frames, 30, 3)
        velocities = (rotation[:, :, None] @ velocities.unsqueeze(-1)).squeeze(-1)
        yaw_velocity = torch.zeros_like(angle)
        if frames > 1:
            delta = angle[:, 1:] - angle[:, :-1]
            yaw_velocity[:, :-1] = torch.atan2(torch.sin(delta), torch.cos(delta)) * self.fps
            yaw_velocity[:, -1] = yaw_velocity[:, -2]
        result = torch.cat([
            height, velocities[:, :, 0, [0, 2]], yaw_velocity.unsqueeze(-1),
            positions.flatten(-2), columns.flatten(-3), velocities.flatten(-2),
            features[..., Global.FOOT_CONTACTS],
        ], dim=-1)
        return result[0] if unbatched else result

    def encode(self, local_rotations, root_positions, canonicalize: bool = True):
        return self(local_rotations, root_positions, to_canonicalize=canonicalize)

    def canonicalize(self, features: torch.Tensor) -> torch.Tensor:
        """Features already discard absolute planar position and heading."""
        return features.clone()

    def inverse(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Reconstruct rotations and FK joints with frame-0 XZ/yaw set to zero."""
        if features.shape[-1] != self.FEATURE_DIM:
            raise ValueError(f"Expected {self.FEATURE_DIM} features, got {features.shape[-1]}.")
        unbatched = features.ndim == 2
        if unbatched:
            features = features.unsqueeze(0)
        batch, frames = features.shape[:2]
        if frames == 0:
            raise ValueError("Motion must contain at least one frame.")
        angle = features.new_zeros(batch, frames)
        angle[:, 1:] = torch.cumsum(
            features[:, :-1, self.ROOT_YAW_VELOCITY].squeeze(-1) / self.fps, dim=1
        )
        rotation = y_rotation(angle)
        root_velocity = features.new_zeros(batch, frames, 3)
        root_velocity[..., [0, 2]] = features[..., self.ROOT_VELOCITY]
        root_velocity = torch.matmul(rotation, root_velocity.unsqueeze(-1)).squeeze(-1)
        root_positions = torch.zeros_like(root_velocity)
        root_positions[:, 1:] = torch.cumsum(root_velocity[:, :-1] / self.fps, dim=1)
        root_positions[..., 1:2] = features[..., self.ROOT_HEIGHT]
        rotations = cont6d_to_matrix(
            features[..., self.LOCAL_ROTATIONS].reshape(batch, frames, 30, 6)
        )
        global_rotations = rotation[:, :, None] @ rotations
        local_rotations = global_rots_to_local_rots(
            global_rotations.reshape(-1, 30, 3, 3), self.skeleton.parents
        ).reshape(batch, frames, 30, 3, 3)
        _, positions, _ = self.skeleton.fk(local_rotations, root_positions)
        output = {
            "local_rot_mats": local_rotations,
            "global_rot_mats": global_rotations,
            "posed_joints": positions,
            "root_positions": root_positions,
            "smooth_root_pos": root_positions,
            "foot_contacts": features[..., self.FOOT_CONTACTS] > 0.5,
            "global_root_heading": torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1),
        }
        return {key: value[0] for key, value in output.items()} if unbatched else output

    def decode(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.inverse(features)


__all__ = ["LocalMotionJEPARep"]
