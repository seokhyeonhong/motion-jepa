"""Coordinate and reconstruction checks for the local representation."""

import unittest

import torch

from motion_rep import LocalMotionJEPARep
from motion_rep.geometry import y_rotation


class LocalMotionJEPARepTests(unittest.TestCase):
    def setUp(self):
        self.rep = LocalMotionJEPARep()
        self.frames = 12
        self.rotations = torch.eye(3).repeat(2, self.frames, 30, 1, 1)
        # Cross the atan2 branch cut while moving and changing root height.
        self.rotations[:, :, 0] = y_rotation(torch.linspace(2.8, 3.6, self.frames))
        self.rotations[:, :, 5] = y_rotation(torch.linspace(0, 0.3, self.frames))
        self.roots = torch.zeros(2, self.frames, 3)
        self.roots[..., 0] = torch.linspace(2, 3, self.frames)
        self.roots[..., 1] = torch.linspace(0.9, 1.1, self.frames)
        self.roots[..., 2] = torch.linspace(-1, 2, self.frames)

    def test_layout_and_root_dynamics(self):
        features = self.rep(self.rotations, self.roots)
        self.assertEqual(features.shape, (2, self.frames, 365))
        torch.testing.assert_close(features[..., self.rep.ROOT_HEIGHT], self.roots[..., 1:2])
        self.assertTrue(torch.isfinite(features).all())
        # Wrapped angular differences must not create a 2*pi spike.
        self.assertLess(features[..., self.rep.ROOT_YAW_VELOCITY].abs().max().item(), 5)
        self.assertGreater(features[..., self.rep.ROOT_VELOCITY].abs().max().item(), 1)

    def test_global_yaw_and_translation_invariance(self):
        features = self.rep(self.rotations, self.roots)
        yaw = y_rotation(torch.tensor(0.7))
        rotations = self.rotations.clone()
        rotations[:, :, 0] = yaw @ rotations[:, :, 0]
        roots = (yaw @ self.roots.unsqueeze(-1)).squeeze(-1)
        roots += torch.tensor([10.0, 0.0, -7.0])
        transformed = self.rep(rotations, roots)
        torch.testing.assert_close(transformed, features, atol=2e-4, rtol=2e-4)

    def test_roundtrip(self):
        features = self.rep(self.rotations, self.roots)
        decoded = self.rep.inverse(features)
        encoded = self.rep(decoded["local_rot_mats"], decoded["root_positions"])
        torch.testing.assert_close(encoded, features, atol=2e-4, rtol=2e-4)
        # Compare reconstructed geometry to canonicalized original geometry.
        _, joints, _ = self.rep.skeleton.fk(self.rotations, self.roots)
        right, left = self.rep.skeleton.hip_joint_idx
        hip = joints[:, 0, right] - joints[:, 0, left]
        yaw = y_rotation(-torch.atan2(hip[..., 2], -hip[..., 0]))
        expected_roots = (yaw[:, None] @ self.roots.unsqueeze(-1)).squeeze(-1)
        expected_joints = (yaw[:, None, None] @ joints.unsqueeze(-1)).squeeze(-1)
        origin = expected_roots[:, :1].clone()
        origin[..., 1] = 0
        expected_roots -= origin
        expected_joints -= origin.unsqueeze(-2)
        torch.testing.assert_close(decoded["root_positions"], expected_roots, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(decoded["posed_joints"], expected_joints, atol=2e-5, rtol=2e-5)

    def test_unbatched_and_single_frame(self):
        features = self.rep.encode(self.rotations[0, :1], self.roots[0, :1])
        self.assertEqual(features.shape, (1, 365))
        torch.testing.assert_close(features[:, 1:4], torch.zeros(1, 3))
        torch.testing.assert_close(features[:, self.rep.VELOCITIES], torch.zeros(1, 90))
        decoded = self.rep.decode(features)
        self.assertEqual(decoded["posed_joints"].shape, (1, 30, 3))
        torch.testing.assert_close(
            self.rep(decoded["local_rot_mats"], decoded["root_positions"]),
            features, atol=2e-5, rtol=2e-5,
        )

    def test_velocity_is_not_derivative_of_local_pose(self):
        rotations = torch.eye(3).repeat(1, 3, 30, 1, 1)
        roots = torch.zeros(1, 3, 3)
        roots[..., 1] = 1
        roots[0, :, 2] = torch.arange(3) / self.rep.fps
        features = self.rep(rotations, roots)
        positions = features[..., self.rep.LOCAL_POSITIONS]
        torch.testing.assert_close(positions[:, 1:], positions[:, :-1])
        velocities = features[..., self.rep.VELOCITIES].reshape(1, 3, 30, 3)
        torch.testing.assert_close(velocities[..., [0, 2]].norm(dim=-1), torch.ones(1, 3, 30))

    def test_mixed_input_dtypes(self):
        for rotation_dtype, root_dtype in ((torch.float64, torch.float32),
                                           (torch.float32, torch.float64)):
            rotations = self.rotations.to(rotation_dtype)
            roots = self.roots.to(root_dtype)
            expected = self.rep(rotations.double(), roots.double())
            actual = self.rep(rotations, roots)
            self.assertEqual(actual.dtype, torch.float64)
            torch.testing.assert_close(actual, expected)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            LocalMotionJEPARep(fps=0)
        with self.assertRaises(ValueError):
            self.rep(self.rotations[:, :0], self.roots[:, :0])
        with self.assertRaises(ValueError):
            self.rep.inverse(torch.zeros(2, 366))


if __name__ == "__main__":
    unittest.main()
