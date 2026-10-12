#include "smpl.h"

#include <torch/csrc/utils/pybind.h>

#include <toast/math/se3.h>
#include <toast/utils/launch_helpers.h>
#include <toast/utils/tensor_helpers.h>

#include "utils/reduce_helpers.cuh"
#include <ATen/ops/zeros_like.h>


__global__ void interpolate_smpl_poses_fwd_kernel(
    const uint num_objects,
    const uint num_joints,
    const uint num_frames,
    const float* query_time_ptr, // [1,]

    const float4* __restrict__ pose_quats, // [num_objects, num_frames, num_joints, 4]
    const float* __restrict__ pose_time,   // [num_objects, num_frames]
    const float4* __restrict__ root_q,     // [num_objects, 4]
    const uint* __restrict__ instance_ids, // [num_objects, ]

    float4* __restrict__ out_quats  // [num_objects, num_joints, 4]
) {
    const uint object_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const uint joint_idx = threadIdx.y;

    if ((object_idx >= num_objects) || (joint_idx >= num_joints)) {
        return;
    }

    uint inst_offset = num_frames * object_idx;
    uint frame_idx = num_frames;
    const float query_time = query_time_ptr[0];
    for (uint i = 0; i < num_frames; i++) {
        if (pose_time[inst_offset + i] >= query_time) {
            frame_idx = i;
            break;
        }
    }

    const uint left_frame_idx = (frame_idx == 0) ? 0 : frame_idx - 1;
    const uint right_frame_idx = min(num_frames - 1, frame_idx);

    const float left_time = pose_time[inst_offset + left_frame_idx];
    const float right_time = pose_time[inst_offset + right_frame_idx];
    const float weight = __saturatef(
        (query_time - left_time) / (right_time - left_time));

    const uint offset = object_idx * num_joints * num_frames + joint_idx;

    quatf left_q = quatf{pose_quats[offset + left_frame_idx * num_joints]}.std();
    quatf right_q = quatf{pose_quats[offset + right_frame_idx * num_joints]}.std();

    out_quats[object_idx * num_joints + joint_idx] =
        (joint_idx == 0) ?
            root_q[instance_ids[object_idx]] : left_q.slerp(right_q, weight).q;
}


at::Tensor interpolate_smpl_poses_fwd(
    const at::Tensor& smpl_poses,  // [N, frames, 24, 4]
    const at::Tensor& smpl_poses_time, // [N, frames]
    const at::Tensor& root_q, // [K, 4]
    const at::Tensor& instance_ids, // [N, ]
    const at::Tensor& query_time
) {
    check_same_device(smpl_poses, smpl_poses_time, root_q, instance_ids, query_time);
    all_contiguous(smpl_poses, smpl_poses_time, root_q, instance_ids, query_time);

    const uint num_objects = smpl_poses.size(0);
    const uint num_frames = smpl_poses.size(1);
    const uint num_joints = smpl_poses.size(2);

    TORCH_CHECK(num_joints == 24, "num_joints only 24 supported now");

    const at::cuda::OptionalCUDAGuard device_guard(smpl_poses.device());

    at::Tensor out_quats = at::empty(
        {num_objects, num_joints, 4}, smpl_poses.options());

    constexpr int objects_per_block = 32;
    dim3 threads(objects_per_block, num_joints, 1);
    uint num_blocks = (num_objects + objects_per_block - 1) / objects_per_block;

    interpolate_smpl_poses_fwd_kernel<<<num_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_objects,
        num_joints,
        num_frames,
        query_time.data_ptr<float>(),
        (float4*)smpl_poses.data_ptr(),
        (float*)smpl_poses_time.data_ptr(),
        (float4*)root_q.data_ptr(),
        (uint*)instance_ids.data_ptr(),
        (float4*)out_quats.data_ptr()
    );

    return out_quats;
}



__global__ void interpolate_smpl_poses_bwd_kernel(
    const uint num_objects,
    const uint num_joints,
    const uint num_frames,
    const float* query_time_ptr,            // [1,]

    const float4* __restrict__ pose_quats,  // [num_objects, num_frames, num_joints, 4]
    const float* __restrict__ pose_time,    // [num_objects, num_frames]
    const uint* __restrict__ instance_ids,  // [num_objects,]

    const float4* __restrict__ grad_out_quats, // [num_objects, num_joints, 4]

    float4* __restrict__ d_pose_quats,      // [num_objects, num_frames, num_joints, 4]
    float4* __restrict__ d_root_q,          // [K, 4]
    float* __restrict__ d_query_time        // [1,]
) {
    const uint object_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const uint joint_idx = threadIdx.y;

    if ((object_idx >= num_objects) || (joint_idx >= num_joints)) {
        return;
    }

    uint inst_offset = num_frames * object_idx;
    uint frame_idx = num_frames;
    const float query_time = query_time_ptr[0];
    for (uint i = 0; i < num_frames; i++) {
        if (pose_time[inst_offset + i] >= query_time) {
            frame_idx = i;
            break;
        }
    }

    const uint left_frame_idx = (frame_idx == 0) ? 0 : frame_idx - 1;
    const uint right_frame_idx = min(num_frames - 1, frame_idx);

    const float left_time = pose_time[inst_offset + left_frame_idx];
    const float right_time = pose_time[inst_offset + right_frame_idx];
    const float dt = right_time - left_time;
    const float raw_weight = (dt > 0.f) ? (query_time - left_time) / dt : 0.5f;
    const float weight = __saturatef(raw_weight);

    const float4 grad_out = grad_out_quats[object_idx * num_joints + joint_idx];

    float d_query_time_local = 0.0f;

    if (joint_idx == 0) {
        const uint inst_id = instance_ids[object_idx];
        atomicAdd(&d_root_q[inst_id].x, grad_out.x);
        atomicAdd(&d_root_q[inst_id].y, grad_out.y);
        atomicAdd(&d_root_q[inst_id].z, grad_out.z);
        atomicAdd(&d_root_q[inst_id].w, grad_out.w);
    } else {
        const uint offset = object_idx * num_joints * num_frames + joint_idx;
        const uint left_idx  = offset + left_frame_idx * num_joints;
        const uint right_idx = offset + right_frame_idx * num_joints;

        const quatf orig_left{pose_quats[left_idx]};
        const quatf orig_right{pose_quats[right_idx]};

        auto [dq1, dq2, dweight] = quatf::slerp_backward(
            orig_left.std().q, orig_right.std().q, weight, grad_out);

        const float4 d_left  = quatf::std_backward(orig_left,  dq1);
        const float4 d_right = quatf::std_backward(orig_right, dq2);

        if (left_idx == right_idx) {
            d_pose_quats[left_idx] = d_left + d_right;
        } else {
            d_pose_quats[left_idx]  = d_left;
            d_pose_quats[right_idx] = d_right;
        }

        if ((raw_weight > 0.0f) && (raw_weight < 1.0f) && (dt > 0.0f)) {
            d_query_time_local = dweight / dt;
        }
    }

    __shared__ cub::BlockReduce<float, 24*32>::TempStorage float_storage;
    float query_time_reduced =
        cub::BlockReduce<float, 24*32>(float_storage).Sum(d_query_time_local);

    if (threadIdx.x == 0) {
        atomicAdd(d_query_time, query_time_reduced);
    }
}


std::tuple<at::Tensor, at::Tensor, at::Tensor> interpolate_smpl_poses_bwd(
    const at::Tensor& smpl_poses,       // [N, frames, 24, 4]
    const at::Tensor& smpl_poses_time,  // [N, frames]
    const at::Tensor& root_q,           // [K, 4]
    const at::Tensor& instance_ids,     // [N,]
    const at::Tensor& query_time,       // [1,]
    const at::Tensor& grad_out_quats    // [N, num_joints, 4]
) {
    check_same_device(smpl_poses, smpl_poses_time, root_q, instance_ids, query_time, grad_out_quats);
    all_contiguous(smpl_poses, smpl_poses_time, root_q, instance_ids, query_time, grad_out_quats);

    const uint num_objects = smpl_poses.size(0);
    const uint num_frames  = smpl_poses.size(1);
    const uint num_joints  = smpl_poses.size(2);

    TORCH_CHECK(num_joints == 24, "num_joints only 24 supported now");

    const at::cuda::OptionalCUDAGuard device_guard(smpl_poses.device());

    at::Tensor d_pose_quats = at::zeros_like(smpl_poses);
    at::Tensor d_root_q     = at::zeros_like(root_q);
    at::Tensor d_query_time = at::zeros_like(query_time);

    constexpr int objects_per_block = 32;
    dim3 threads(objects_per_block, num_joints, 1);
    uint num_blocks = (num_objects + objects_per_block - 1) / objects_per_block;

    interpolate_smpl_poses_bwd_kernel
        <<<num_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_objects,
        num_joints,
        num_frames,
        query_time.data_ptr<float>(),
        (float4*)smpl_poses.data_ptr(),
        (float*)smpl_poses_time.data_ptr(),
        (uint*)instance_ids.data_ptr(),
        (float4*)grad_out_quats.data_ptr(),
        (float4*)d_pose_quats.data_ptr(),
        (float4*)d_root_q.data_ptr(),
        (float*)d_query_time.data_ptr()
    );

    return {d_pose_quats, d_root_q, d_query_time};
}



void bind_smpl_interpolate(py::module_ &m) {
    m.def("interpolate_smpl_poses_fwd", &interpolate_smpl_poses_fwd,
        py::arg("smpl_poses"), py::arg("smpl_poses_time"),
        py::arg("root_q"), py::arg("instance_ids"),
        py::arg("query_time"));

    m.def("interpolate_smpl_poses_bwd", &interpolate_smpl_poses_bwd,
        py::arg("smpl_poses"), py::arg("smpl_poses_time"),
        py::arg("root_q"), py::arg("instance_ids"),
        py::arg("query_time"), py::arg("grad_out_quats"));
}