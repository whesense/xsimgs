import numpy as np
import torch

from xsimgs._C import (
    copy_gs_rigid_transform_fwd,
    copy_gs_rigid_transform_bwd,
    packed_mask_from_instances
)

from toast import Quat, SE3
from toast.ops.ext import apply_grad_mask


class RigidTransformAndPut(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            positions: torch.Tensor,
            rotations: torch.Tensor,
            instance_id: torch.Tensor,
            inst_q: torch.Tensor,
            inst_t: torch.Tensor,
            inst_v: torch.Tensor,
            inst_w: torch.Tensor,
            inst_mask: torch.Tensor,

            out_positions: torch.Tensor,
            out_rotation: torch.Tensor,
            out_velocity: torch.Tensor,
            out_mask: torch.Tensor,

            buffer_offset: int,
            per_instance_offsets: torch.Tensor,
            gs_inst_offsets: torch.Tensor
    ):
        copy_gs_rigid_transform_fwd(
            positions, rotations, instance_id,
            inst_q, inst_t, inst_v, inst_w, inst_mask,
            out_positions, out_rotation, out_velocity, out_mask,
            buffer_offset, per_instance_offsets, gs_inst_offsets
        )
        ctx.buffer_offset = buffer_offset
        ctx.save_for_backward(
            positions, rotations, instance_id,
            inst_q, inst_w, inst_mask,
            per_instance_offsets,
            gs_inst_offsets
        )

        return out_positions, out_rotation, out_velocity

    @staticmethod
    def backward(
            ctx,
            v_positions: torch.Tensor,
            v_rotation: torch.Tensor,
            v_velocity: torch.Tensor
    ):
        (
            positions, rotations, instance_id,
            inst_q, inst_w, inst_mask,
            per_instance_offsets,
            gs_inst_offsets
        ) = ctx.saved_tensors

        (
            grad_positions, grad_rotations,
            grad_q, grad_t, grad_v, grad_w
        ) = copy_gs_rigid_transform_bwd(
            positions,
            rotations,
            instance_id,
            inst_q,
            inst_w,
            inst_mask,
            ctx.buffer_offset,
            per_instance_offsets,
            gs_inst_offsets,
            v_positions,
            v_rotation,
            v_velocity
        )
        return apply_grad_mask(ctx, (
            grad_positions, grad_rotations, None,
            grad_q, grad_t, grad_v, grad_w, None,
            v_positions, v_rotation, v_velocity, None,
            None, None, None
        ))


def rigid_transform_and_put(
        positions: torch.Tensor,    # [N, 3]
        rotations: torch.Tensor,    # [N, 4]
        instance_id: torch.Tensor,  # [N,]

        inst_q: torch.Tensor,       # [K, 4]
        inst_t: torch.Tensor,       # [K, 3]
        inst_v: torch.Tensor,       # [K, 3]
        inst_w: torch.Tensor,       # [K, 3]
        inst_mask: torch.Tensor,    # [K, ]

        out_positions: torch.Tensor = None,# [B, 3]
        out_rotation: torch.Tensor = None, # [B, 4]
        out_velocity: torch.Tensor = None, # [B, 3]
        out_mask: torch.Tensor = None,     # [B,]

        buffer_offset: int = 0,
        per_instance_offsets: torch.Tensor = None, # [K,]
        gs_inst_offsets: torch.Tensor = None      # [N,]
):
    if out_positions is None:
        out_positions = torch.zeros_like(positions.detach())
    if out_rotation is None:
        out_rotation = torch.zeros_like(rotations.detach())
    if out_velocity is None:
        out_velocity = torch.zeros_like(positions.detach())
    if out_mask is None:
        out_mask = torch.zeros_like(positions.detach()[..., 0], dtype=torch.bool)

    out_positions, out_rotation, out_velocity = RigidTransformAndPut.apply(
        positions,
        rotations,
        instance_id,
        inst_q,
        inst_t,
        inst_v,
        inst_w,
        inst_mask,

        out_positions,
        out_rotation,
        out_velocity,
        out_mask,

        buffer_offset,
        per_instance_offsets,
        gs_inst_offsets
    )
    return out_positions, out_rotation, out_velocity, out_mask


def rigid_transform_and_put_torch(
        positions: torch.Tensor,    # [N, 3]
        rotations: torch.Tensor,    # [N, 4]
        instance_id: torch.Tensor,  # [N,]

        inst_q: torch.Tensor,       # [K, 4]
        inst_t: torch.Tensor,       # [K, 3]
        inst_v: torch.Tensor,       # [K, 3]
        inst_w: torch.Tensor,       # [K, 3]
        inst_mask: torch.Tensor,    # [K, ]

        out_positions: torch.Tensor,       # [B, 3]
        out_rotation: torch.Tensor,        # [B, 4]
        out_velocity: torch.Tensor | None, # [B, 3]
        out_mask: torch.Tensor | None,     # [B,]

        buffer_offset: int,
        per_instance_offsets: torch.Tensor | None = None, # [K,]
        gs_inst_offsets: torch.Tensor | None = None       # [N,]
):
    if out_positions is None:
        out_positions = torch.zeros_like(positions.detach())
        out_rotation = torch.zeros_like(rotations.detach())
        out_velocity = torch.zeros_like(positions.detach())
        out_mask = torch.zeros_like(positions.detach()[..., 0], dtype=torch.bool)

    if (per_instance_offsets is not None) and (gs_inst_offsets is not None):
        out_slice = buffer_offset + gs_inst_offsets + per_instance_offsets[instance_id]
    else:
        out_slice = torch.arange(
            buffer_offset, buffer_offset + len(positions),
            device=positions.device
        )

    input_mask = inst_mask[instance_id].nonzero().view(-1)
    out_slice = out_slice[input_mask]
    instance_id_masked = instance_id[input_mask]

    inst_pose = SE3(inst_q, inst_t)
    gs_poses = inst_pose[instance_id[input_mask]]
    positions_masked = positions[input_mask]
    out_positions[out_slice] = gs_poses @ positions_masked
    out_rotation[out_slice] = (gs_poses.q @ Quat(rotations[input_mask])).data

    if out_velocity is not None:
        lever = torch.cross(
            inst_w[instance_id_masked],
            gs_poses.q @ positions_masked,
            dim=-1
        )
        out_velocity[out_slice] = inst_v[instance_id_masked] + lever

    if out_mask is not None:
        out_mask[out_slice] = True

    return out_positions, out_rotation, out_velocity, out_mask
