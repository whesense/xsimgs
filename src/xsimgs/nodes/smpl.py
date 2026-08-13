import torch
from toast.ops.ext import apply_grad_mask

from xsimgs._C import (
    interpolate_smpl_poses_fwd,
    interpolate_smpl_poses_bwd,

    smpl_local_to_global_fwd,
    smpl_local_to_global_bwd,

    smpl_lbs_skinning as smpl_lbs_skinning_fwd,
    smpl_lbs_skinning_bwd,

    smpl_ids_compaction
)


class InterpolateSMPLPoses(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            smpl_poses: torch.Tensor, # [num_objects, num_keyframes, num_joints, 4]
            smpl_poses_time: torch.Tensor, # [num_objects, num_keyframes],
            root_q: torch.Tensor, # [num_objects, 4]
            instance_ids: torch.Tensor, # [num_objects, ]
            query_time: torch.Tensor # [1, ]
    ) -> torch.Tensor:
        ctx.save_for_backward(
            smpl_poses, smpl_poses_time, root_q, instance_ids, query_time)
        return interpolate_smpl_poses_fwd(
            smpl_poses, smpl_poses_time, root_q, instance_ids, query_time
        )

    @staticmethod
    def backward(ctx, grad_quat_out: torch.Tensor):
        (
            smpl_poses, smpl_poses_time, root_q, instance_ids, query_time
        ) = ctx.saved_tensors
        grad_pose_quats, grad_root_q, grad_query = interpolate_smpl_poses_bwd(
            smpl_poses, smpl_poses_time, root_q, instance_ids, query_time,
            grad_quat_out.contiguous()
        )
        return apply_grad_mask(
            ctx, [grad_pose_quats, None, grad_root_q, None, grad_query])


class SMPLLocalToGlobalPose(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            local_q_ids: torch.Tensor,
            local_q: torch.Tensor,
            joints: torch.Tensor,
            a0_inv_q: torch.Tensor,
            a0_inv_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out_q, out_t = smpl_local_to_global_fwd(
            local_q_ids, local_q, joints, a0_inv_q, a0_inv_t
        )
        ctx.save_for_backward(
            local_q_ids, local_q, joints, a0_inv_q, a0_inv_t,
            out_q, out_t
        )
        return out_q, out_t

    @staticmethod
    def backward(
            ctx,
            grad_q: torch.Tensor,
            grad_t: torch.Tensor
    ):
        (
            local_q_ids, local_q, joints, a0_inv_q, a0_inv_t, out_q, out_t
        ) = ctx.saved_tensors
        grad_local_q, grad_a0_q, grad_a0_t = smpl_local_to_global_bwd(
            local_q_ids, local_q, joints, a0_inv_q, a0_inv_t, out_q, out_t,
            grad_q.contiguous(), grad_t.contiguous()
        )
        return apply_grad_mask(ctx, [None, grad_local_q, None, grad_a0_q, grad_a0_t])


class LBSSkinningSMPL(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            valid_ids: torch.Tensor,      # [B,]
            global_q: torch.Tensor,       # [B, 24, 4]
            global_t: torch.Tensor,       # [B, 24, 3]
            vertices: torch.Tensor,       # [N, V, 3]
            quats: torch.Tensor,          # [N, V, 4]
            weights: torch.Tensor,        # [N, 24, V]
            instance_ids: torch.Tensor,   # [N,]
            instance_t: torch.Tensor,     # [K, 3]
            instance_v: torch.Tensor,     # [K, 3]
            global_offset: int,
            out_positions: torch.Tensor,  # [container_size, 3]
            out_rotations: torch.Tensor,  # [container_size, 4]
            out_velocities: torch.Tensor, # [container_size, 3]
            out_mask: torch.Tensor,       # [container_size, ]
    ):
        smpl_lbs_skinning_fwd(
            valid_ids, global_q, global_t,
            vertices, quats, weights,
            instance_ids, instance_t, instance_v,
            global_offset,
            out_positions, out_rotations, out_velocities, out_mask
        )
        ctx.global_offset = global_offset
        ctx.num_instances = instance_t.shape[0]
        ctx.save_for_backward(
            valid_ids, instance_ids,
            global_q, global_t, vertices, quats, weights
        )
        return out_positions, out_rotations, out_velocities, out_mask

    @staticmethod
    def backward(
            ctx,
            v_positions: torch.Tensor,
            v_rotations: torch.Tensor,
            v_velocities: torch.Tensor,
            v_mask: torch.Tensor
    ):
        (
            valid_ids, instance_ids,
            global_q, global_t, vertices, quats, weights
        ) = ctx.saved_tensors

        v_positions = v_positions.contiguous()
        v_rotations = v_rotations.contiguous()
        v_velocities = (
            v_velocities.contiguous() if v_velocities is not None else None
        )

        (
            d_global_q, d_global_t,
            d_vertices, d_quats, d_weights,
            d_instance_v, d_instance_t
        ) = smpl_lbs_skinning_bwd(
            valid_ids, instance_ids,
            global_q, global_t, vertices, quats, weights,
            ctx.global_offset, ctx.num_instances,
            v_positions, v_rotations, v_velocities
        )

        return apply_grad_mask(ctx, (
            None,          # valid_ids
            d_global_q,    # global_q
            d_global_t,    # global_t
            d_vertices,    # vertices
            d_quats,       # quats
            d_weights,     # weights
            None,          # instance_ids
            d_instance_t,  # instance_t
            d_instance_v,  # instance_v
            None,          # global_offset
            v_positions,   # out_positions (passthrough)
            v_rotations,   # out_rotations (passthrough)
            v_velocities,  # out_velocities (passthrough)
            None           # out_mask
        ))


def smpl_lbs_skinning(
        valid_ids: torch.Tensor,
        global_q: torch.Tensor,
        global_t: torch.Tensor,
        vertices: torch.Tensor,
        quats: torch.Tensor,
        weights: torch.Tensor,
        instance_ids: torch.Tensor,
        instance_t: torch.Tensor,
        instance_v: torch.Tensor,
        global_offset: int,
        out_positions: torch.Tensor,
        out_rotations: torch.Tensor,
        out_velocities: torch.Tensor | None = None,
        out_mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    out_p, out_r, out_v, out_m = LBSSkinningSMPL.apply(
        valid_ids.contiguous(),
        global_q.contiguous(), global_t.contiguous(),
        vertices.contiguous(), quats.contiguous(), weights.contiguous(),
        instance_ids.contiguous(),
        instance_t.contiguous(), instance_v.contiguous(),
        global_offset,
        out_positions, out_rotations, out_velocities, out_mask
    )
    return out_p, out_r, out_v, out_m


def smpl_local_to_global_pose(
        local_q_ids: torch.Tensor,
        local_q: torch.Tensor,
        joints: torch.Tensor,
        a0_inv_q: torch.Tensor,
        a0_inv_t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return SMPLLocalToGlobalPose.apply(
        local_q_ids.contiguous(), local_q.contiguous(),
        joints.contiguous(),
        a0_inv_q.contiguous(), a0_inv_t.contiguous()
    )


def interpolate_smpl_poses(
        smpl_poses: torch.Tensor, # [num_objects, num_keyframes, num_joints, 4]
        smpl_poses_time: torch.Tensor, # [num_objects, num_keyframes],
        root_q: torch.Tensor, # [num_objects, 4]
        instance_ids: torch.Tensor, # [num_objects, ]
        query_time: torch.Tensor # [1, ]
) -> torch.Tensor:  # [num_objects, num_joints, 4]
    return InterpolateSMPLPoses.apply(
        smpl_poses.contiguous(), smpl_poses_time.contiguous(),
        root_q.contiguous(), instance_ids.contiguous(),
        query_time
    )



def valid_instances_indices(
        instance_ids: torch.Tensor,  # [N, ]
        instance_mask: torch.Tensor, # [K, ]
):
    if (
            len(instance_ids) < 1024 and
            instance_ids.is_contiguous() and
            instance_mask.is_contiguous()
    ):
        return smpl_ids_compaction(instance_ids, instance_mask)

    idxs = torch.arange(len(instance_ids), device=instance_ids.device)
    return idxs[instance_mask[instance_ids]]
