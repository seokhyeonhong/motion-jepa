"""Skeleton rotation transforms."""

import torch

from .kinematics import global_rots_to_local_rots, local_rots_to_global_rots


def to_standard_tpose(
    local_rotations: torch.Tensor,
    parents: torch.Tensor,
    global_rotation_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = global_rotation_offsets.to(
        device=local_rotations.device, dtype=local_rotations.dtype
    )
    global_rotations = local_rots_to_global_rots(local_rotations, parents)
    transformed_global = torch.einsum(
        "tnij,nkj->tnik", global_rotations, offsets
    )
    return (
        global_rots_to_local_rots(transformed_global, parents),
        transformed_global,
    )


__all__ = ["global_rots_to_local_rots", "to_standard_tpose"]
