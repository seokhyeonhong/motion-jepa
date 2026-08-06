# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shaded sphere-and-cylinder skeleton rendering for viser."""

from __future__ import annotations

import numpy as np
import trimesh

from skeleton import SOMASkeleton77

JOINT_COLOR = np.asarray((255, 235, 0), dtype=np.uint8)
BONE_COLOR = np.asarray((27, 106, 0), dtype=np.uint8)
CONTACT_COLOR = np.asarray((255, 0, 0), dtype=np.uint8)


def bone_transforms(
    joints: np.ndarray,
    parents: np.ndarray,
    *,
    epsilon: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return midpoint, length, and +Z-to-bone quaternion for each non-root joint."""
    joints = np.asarray(joints, dtype=np.float32)
    parents = np.asarray(parents, dtype=np.int64)
    starts = joints[parents[1:]]
    ends = joints[1:]
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=-1)
    midpoints = (starts + ends) * 0.5

    # A unit quaternion rotating +Z onto the bone direction is
    # normalize([1 + dot(z, direction), cross(z, direction)]).
    quaternions = np.zeros((len(vectors), 4), dtype=np.float32)
    quaternions[:, 0] = 1.0
    valid = lengths > epsilon
    directions = np.zeros_like(vectors)
    directions[valid] = vectors[valid] / lengths[valid, None]
    quaternions[valid, 0] = 1.0 + directions[valid, 2]
    quaternions[valid, 1] = -directions[valid, 1]
    quaternions[valid, 2] = directions[valid, 0]
    quaternions[valid, 3] = 0.0

    antiparallel = valid & (quaternions[:, 0] < epsilon)
    quaternions[antiparallel] = (0.0, 1.0, 0.0, 0.0)
    norms = np.linalg.norm(quaternions, axis=-1)
    quaternions /= np.maximum(norms[:, None], epsilon)
    return midpoints, lengths.astype(np.float32), quaternions


class ShadedSkeletonRenderer:
    """A reusable, batched viser skeleton made from shaded triangle meshes."""

    def __init__(
        self,
        scene,
        joints: np.ndarray,
        *,
        skeleton: SOMASkeleton77 | None = None,
        visible: bool = True,
        joint_radius: float = 0.025,
        finger_joint_radius: float = 0.015,
        bone_radius: float = 0.012,
    ):
        self.scene = scene
        self.skeleton = skeleton or SOMASkeleton77()
        self.parents = self.skeleton.parents.detach().cpu().numpy()
        self.joint_radius = float(joint_radius)
        self.finger_joint_radius = float(finger_joint_radius)
        self.bone_radius = float(bone_radius)
        self.joint_scales = np.asarray(
            [
                finger_joint_radius
                if ("Hand" in name and not name.endswith("Hand"))
                else joint_radius
                for name in self.skeleton.names
            ],
            dtype=np.float32,
        )
        self.normal_joint_colors = np.tile(JOINT_COLOR, (len(self.parents), 1))

        sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        cylinder = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=12)
        positions, lengths, orientations = bone_transforms(joints, self.parents)
        identity = np.zeros((len(self.parents), 4), dtype=np.float32)
        identity[:, 0] = 1.0
        self.joints = scene.add_batched_meshes_simple(
            "/motion_jepa/joints",
            vertices=np.asarray(sphere.vertices, dtype=np.float32),
            faces=np.asarray(sphere.faces, dtype=np.int32),
            batched_wxyzs=identity,
            batched_positions=np.asarray(joints, dtype=np.float32),
            batched_scales=self.joint_scales,
            batched_colors=self.normal_joint_colors,
            material="standard",
            cast_shadow=True,
            receive_shadow=True,
            visible=visible,
        )
        self.bones = scene.add_batched_meshes_simple(
            "/motion_jepa/bones",
            vertices=np.asarray(cylinder.vertices, dtype=np.float32),
            faces=np.asarray(cylinder.faces, dtype=np.int32),
            batched_wxyzs=orientations,
            batched_positions=positions,
            batched_scales=np.column_stack(
                (
                    np.full_like(lengths, bone_radius),
                    np.full_like(lengths, bone_radius),
                    lengths,
                )
            ),
            batched_colors=BONE_COLOR,
            material="standard",
            cast_shadow=True,
            receive_shadow=True,
            visible=visible,
        )

    @property
    def visible(self) -> bool:
        return bool(self.joints.visible)

    @visible.setter
    def visible(self, value: bool) -> None:
        self.joints.visible = bool(value)
        self.bones.visible = bool(value)

    def update(
        self,
        joints: np.ndarray,
        *,
        contact_indices: np.ndarray | None = None,
    ) -> None:
        joints = np.asarray(joints, dtype=np.float32)
        positions, lengths, orientations = bone_transforms(joints, self.parents)
        self.joints.batched_positions = joints
        colors = self.normal_joint_colors.copy()
        if contact_indices is not None:
            colors[np.asarray(contact_indices, dtype=np.int64)] = CONTACT_COLOR
        self.joints.batched_colors = colors
        self.bones.batched_positions = positions
        self.bones.batched_wxyzs = orientations
        self.bones.batched_scales = np.column_stack(
            (
                np.full_like(lengths, self.bone_radius),
                np.full_like(lengths, self.bone_radius),
                lengths,
            )
        )

    def remove(self) -> None:
        self.joints.remove()
        self.bones.remove()


__all__ = [
    "BONE_COLOR",
    "CONTACT_COLOR",
    "JOINT_COLOR",
    "ShadedSkeletonRenderer",
    "bone_transforms",
]
