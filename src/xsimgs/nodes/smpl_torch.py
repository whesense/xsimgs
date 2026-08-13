import torch

from toast import Quat, SE3, sample_trajectory


def interpolate_smpl_poses_torch(
        smpl_poses: torch.Tensor, # [num_objects, num_keyframes, num_joints, 4]
        smpl_poses_time: torch.Tensor, # [num_objects, num_keyframes],
        root_q: torch.Tensor, # [num_objects, 4]
        instance_ids: torch.Tensor, # [num_objects, ]
        query_time: torch.Tensor # [1, ]
) -> torch.Tensor:  # [num_objects, num_joints, 4]
    num_objects, num_keyframes, num_joints, _ = smpl_poses.shape

    q = query_time.view(1, 1, 1).expand(num_objects, num_joints, -1)
    seq_t = smpl_poses_time.unsqueeze(-2) # [num_objects, 1, num_keyframes]
    seq_t = seq_t.expand(-1, num_joints, -1)
    seq_q = Quat(smpl_poses.permute(0, 2, 1, 3)).std().data
    results = sample_trajectory(
        query_times=q, seq_times=seq_t, seq_quats=seq_q, return_indices=True)
    out_quats = results[0][:, :, 0]
    out_quats[:, 0] = root_q[instance_ids]
    return out_quats



def smpl_local_to_global_pose_torch(
        local_q_ids: torch.Tensor,
        local_q: torch.Tensor,
        joints: torch.Tensor,
        a0_inv_q: torch.Tensor,
        a0_inv_t: torch.Tensor
):
    parents = torch.tensor(
        [-1,  0,  0,  0,  1,  2,  3,  4,
         5,  6,  7,  8,  9,  9,  9, 12,
         13, 14, 16, 17, 18, 19, 20, 21
         ], device=joints.device)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    trf = SE3(local_q, rel_joints).std()
    transform_chain = [trf[:, 0]]  # [0] = [B, 4, 4]

    for i in range(1, len(parents)):
        transform_chain.append(transform_chain[parents[i]] @ trf[:, i])

    trf: SE3 = SE3.stack(transform_chain, dim=1)
    tres = SE3(trf.q, trf.t - trf.q @ joints)
    out_trf = tres @ SE3(a0_inv_q, a0_inv_t)
    return out_trf.q.data[local_q_ids], out_trf.t[local_q_ids]



def smpl_lbs_skinning_torch(
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
):
    # print('B =', len(valid_ids))
    valid_weights = weights[valid_ids]  # [B, 24, V]
    valid_weights = valid_weights[..., None, None]  # [B, 24, V, 1, 1]
    # print('valid_weights:', valid_weights.shape)
    # joint_mtx: [B, 24, 1, 3, 4]
    joint_mtx = SE3(global_q, global_t).to_matrix(to_3x4=True).unsqueeze(-3)
    # print('joint_mtx:', joint_mtx.shape)

    vert_mtx = (valid_weights * joint_mtx).sum(dim=1)  # [B, V, 3, 4]
    # print('vert_mtx:', vert_mtx.shape)

    valid_vertices = vertices[valid_ids]  # [B, V, 3]
    # print('valid_vertices:', valid_vertices.shape)
    # [B, V, 4]
    valid_vertices_homo = torch.cat([
        valid_vertices,
        torch.ones_like(valid_vertices[..., 0:1])
    ], dim=-1)
    # print('valid_vertices_homo:', valid_vertices_homo.shape)

    # [B, V, 3, 4] @ [B, V, 4, 1] -> [B, V, 3, 1] -> [B, V, 3]
    result_vertices = (vert_mtx @ valid_vertices_homo.unsqueeze(-1)).squeeze(-1)
    # print('result_vertices:', result_vertices.shape)
    result_vertices = result_vertices.view(-1, 3)

    V = vertices.shape[1]
    out_slice = (valid_ids * V + global_offset).repeat_interleave(V, dim=0)
    out_slice += torch.arange(V, device=valid_ids.device).repeat(len(valid_ids))
    # out_slice = np.s_[global_offset:global_offset + len(result_vertices)]

    inst_indices = instance_ids[valid_ids]
    valid_inst_t = instance_t[inst_indices]  # [B, 3]
    valid_inst_v = instance_v[inst_indices]  # [B, 3]

    # print('V =', V)
    t = valid_inst_t.unsqueeze(1).expand(-1, vertices.shape[1], -1).reshape(-1, 3)
    # print('t=', t.shape)
    out_positions[out_slice] = result_vertices + t

    lbs_quat = Quat.from_matrix(vert_mtx[..., :3, :3]).std()  # [B, V, 4]
    # print('lbs_quat:', lbs_quat.shape)
    gs_quat = Quat(quats.view(-1, V, 4)[valid_ids])
    # print('gs_quat:', gs_quat.shape)
    out_rot = (lbs_quat @ gs_quat).data
    # print('lbs_composed_quats:', out_rot.shape)
    out_rotations[out_slice] = out_rot.view(-1, 4)

    if out_velocities is not None:
        vel = valid_inst_v.unsqueeze(1).expand(-1, vertices.shape[1], -1).reshape(-1, 3)
        out_velocities[out_slice] = vel

    if out_mask is not None:
        out_mask[out_slice] = True

    return out_positions, out_rotations, out_velocities, out_mask
