"""Foot-contact extraction for motion representations."""

import torch


def foot_detect_from_pos_and_vel(
    positions: torch.Tensor,
    velocity: torch.Tensor,
    skeleton,
    vel_thres: float,
    height_thresh: float,
) -> torch.Tensor:
    left = skeleton.left_foot_joint_idx
    right = skeleton.right_foot_joint_idx
    left_contacts = (
        (torch.linalg.norm(velocity[:, :, left], dim=-1) < vel_thres)
        & (positions[:, :, left, 1] < height_thresh)
    )
    right_contacts = (
        (torch.linalg.norm(velocity[:, :, right], dim=-1) < vel_thres)
        & (positions[:, :, right, 1] < height_thresh)
    )
    return torch.cat([left_contacts, right_contacts], dim=-1).to(positions.dtype)
