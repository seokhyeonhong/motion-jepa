"""Tests for the local shaded skeleton renderer."""

from __future__ import annotations

import unittest

import numpy as np

from skeleton import SOMASkeleton77
from visualization import ShadedSkeletonRenderer, bone_transforms


class _Handle:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.removed = False

    def remove(self):
        self.removed = True


class _Scene:
    def __init__(self):
        self.calls = []

    def add_batched_meshes_simple(self, name, **kwargs):
        self.calls.append((name, kwargs))
        return _Handle(
            visible=kwargs["visible"],
            batched_positions=kwargs["batched_positions"],
            batched_wxyzs=kwargs["batched_wxyzs"],
            batched_scales=kwargs["batched_scales"],
            batched_colors=kwargs["batched_colors"],
        )


class ShadedSkeletonTest(unittest.TestCase):
    def test_bone_transforms_and_degenerate_bones(self):
        joints = np.asarray(
            [[0, 0, 0], [0, 0, 2], [0, 0, 2], [0, 0, 1]], dtype=np.float32
        )
        parents = np.asarray([0, 0, 1, 2])
        midpoints, lengths, quaternions = bone_transforms(joints, parents)
        np.testing.assert_allclose(midpoints, [[0, 0, 1], [0, 0, 2], [0, 0, 1.5]])
        np.testing.assert_allclose(lengths, [2, 0, 1])
        np.testing.assert_allclose(quaternions[0], [1, 0, 0, 0])
        np.testing.assert_allclose(quaternions[1], [1, 0, 0, 0])
        np.testing.assert_allclose(quaternions[2], [0, 1, 0, 0])
        self.assertTrue(np.isfinite(quaternions).all())

    def test_finger_scaling_contacts_and_reset(self):
        skeleton = SOMASkeleton77()
        joints = skeleton.neutral_joints.detach().cpu().numpy()
        scene = _Scene()
        renderer = ShadedSkeletonRenderer(scene, joints, skeleton=skeleton)
        hand = skeleton.name_to_index["LeftHand"]
        finger = skeleton.name_to_index["LeftHandIndex1"]
        self.assertAlmostEqual(renderer.joint_scales[hand], renderer.joint_radius)
        self.assertAlmostEqual(
            renderer.joint_scales[finger], renderer.finger_joint_radius
        )

        renderer.update(joints, contact_indices=np.asarray([69, 70]))
        np.testing.assert_array_equal(renderer.joints.batched_colors[69], [255, 0, 0])
        renderer.update(joints, contact_indices=None)
        np.testing.assert_array_equal(renderer.joints.batched_colors[69], [255, 235, 0])


if __name__ == "__main__":
    unittest.main()
