#include "smpl.h"

#include <toast/math/se3.h>
#include <toast/utils/launch_helpers.h>
#include <toast/utils/tensor_helpers.h>

#include "utils.cuh"
#include <ATen/ops/zeros_like.h>



DEVICE quatf shuffle_quat(const uint mask, const quatf src, const int lane_idx) {
    return quatf(float4{
        __shfl_sync(mask, src.q.x, lane_idx),
        __shfl_sync(mask, src.q.y, lane_idx),
        __shfl_sync(mask, src.q.z, lane_idx),
        __shfl_sync(mask, src.q.w, lane_idx)
    });
}

DEVICE float3 shuffle_float3(const uint mask, const float3 src, const int lane_idx) {
    float3 result;
    result.x = __shfl_sync(mask, src.x, lane_idx);
    result.y = __shfl_sync(mask, src.y, lane_idx);
    result.z = __shfl_sync(mask, src.z, lane_idx);
    return result;
}

__global__ void smpl_local_to_global_fwd_kernel(
    const uint batch_size,

    const uint* __restrict__ local_q_ids, // [B]
    const float4* __restrict__ local_q,  // [N, 24, 4]
    const float3* __restrict__ joints,  // [N, 24, 3]

    const float4* __restrict__ a0_inv_q, // [N, 24, 4]
    const float3* __restrict__ a0_inv_t, // [N, 24, 3]

    float4* __restrict__ output_q, // [B, 24, 4]
    float3* __restrict__ output_t // [B, 24, 3]
) {
    constexpr int PARENTS[NUM_JOINTS] = {
        -1,  0,  0,  0,  1,  2,  3,  4,
         5,  6,  7,  8,  9,  9,  9, 12,
        13, 14, 16, 17, 18, 19, 20, 21
    };
    constexpr uint MAX_DEPTH = 8;
    constexpr uint ACTIVE_THREADS = 0xFFFFFF;

    const uint cur_joint = threadIdx.x;
    const uint cur_object_in_batch = blockDim.y * blockIdx.x + threadIdx.y;
    const uint cur_object_id = local_q_ids[cur_object_in_batch];

    if ((cur_joint >= NUM_JOINTS) || (cur_object_in_batch >= batch_size)) {
        return;
    }

    const uint full_offset = cur_object_id * NUM_JOINTS + cur_joint;
    const uint batch_offset = cur_object_in_batch * NUM_JOINTS + cur_joint;

    int next_joint = PARENTS[cur_joint];
    const float3 cur_joint_pos = joints[full_offset];
    const float3 parent_joint_pos = shuffle_float3(
        ACTIVE_THREADS, cur_joint_pos, max(next_joint, 0));
    const float3 rel_joint =
        (cur_joint == 0) ? cur_joint_pos : cur_joint_pos - parent_joint_pos;

    const se3f cur_trf(quatf{local_q[full_offset]}.std(), rel_joint);
    const se3f a0_inv{quatf{a0_inv_q[full_offset]}, a0_inv_t[full_offset]};
    se3f result(cur_trf);

    #pragma unroll
    for (uint i = 0; i < MAX_DEPTH; ++i) {
        // load parent transform
        const int next_joint_valid = max(next_joint, 0);
        se3f parent_trf{
            shuffle_quat(ACTIVE_THREADS, cur_trf.q, next_joint_valid),
            shuffle_float3(ACTIVE_THREADS, cur_trf.t, next_joint_valid)
        };
        if (next_joint != -1) {
            result = parent_trf.compose(result);
            next_joint = PARENTS[next_joint];
        }
    }

    result.t -= result.q.apply(cur_joint_pos);
    result = result.compose(a0_inv);

    output_q[batch_offset] = result.q.q;
    output_t[batch_offset] = result.t;
}


std::tuple<at::Tensor, at::Tensor> smpl_local_to_global_fwd(
    const at::Tensor& local_q_ids,  // [B,]
    const at::Tensor& local_q,  // [B, 24, 4]
    const at::Tensor& joints,  // [N, 24, 3]
    const at::Tensor& a0_inv_q, // [N, 24, 4]
    const at::Tensor& a0_inv_t  // [N, 24, 3]
) {
    const at::cuda::OptionalCUDAGuard device_guard(local_q.device());

    const uint batch_size = local_q_ids.numel();

    at::Tensor result_q = at::empty(
        {batch_size, NUM_JOINTS, 4}, local_q.options());
    at::Tensor result_t = at::empty(
        {batch_size, NUM_JOINTS, 3}, local_q.options());

    constexpr uint num_objects_per_block = 16;
    dim3 threads(32, num_objects_per_block, 1);
    uint num_blocks =
        (batch_size + num_objects_per_block - 1) / num_objects_per_block;

    smpl_local_to_global_fwd_kernel<<<num_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        batch_size, (uint*)local_q_ids.data_ptr(),
        (float4*)local_q.data_ptr(),
        (float3*)joints.data_ptr(),
        (float4*)a0_inv_q.data_ptr(),
        (float3*)a0_inv_t.data_ptr(),
        (float4*)result_q.data_ptr(),
        (float3*)result_t.data_ptr()
    );
    return {result_q, result_t};
}





__global__ void smpl_local_to_global_bwd_kernel(
    const uint batch_size,
    const uint* __restrict__ local_q_ids,     // [B]
    const float4* __restrict__ local_q,       // [N, 24, 4]
    const float3* __restrict__ joints,        // [N, 24, 3]
    const float4* __restrict__ a0_inv_q,      // [N, 24, 4]
    const float3* __restrict__ a0_inv_t,      // [N, 24, 3]

    const float4* __restrict__ output_q,
    const float3* __restrict__ output_t,

    const float4* __restrict__ grad_output_q, // [B, 24, 4]
    const float3* __restrict__ grad_output_t, // [B, 24, 3]
    float4* __restrict__ d_local_q,           // [N, 24, 4]
    float4* __restrict__ d_a0_inv_q,
    float3* __restrict__ d_a0_inv_t
) {
    constexpr int PARENTS[NUM_JOINTS] = {
        -1,  0,  0,  0,  1,  2,  3,  4,
         5,  6,  7,  8,  9,  9,  9, 12,
        13, 14, 16, 17, 18, 19, 20, 21
    };
    constexpr int root_to_leaf_paths[29] = {
        //  0  1  2  3
        1, 4, 7, 10,
    //  4  5  6  7
        2, 5, 8, 11,
    //  8  9  10 11  12  13  14  15
        3, 6, 9, 13, 16, 18, 20, 22,
    //  16 17 18 19  20  21  22  23
        3, 6, 9, 14, 17, 19, 21, 23,
    // 24 25 26  27  28
        3, 6, 9, 12, 15
    };

    constexpr int joint_path_indices[24] = {
        //  0   1  2  3  4  5  6  7  8
        -1, 0, 4, 8, 0, 4, 8, 0, 4,
    //  9  10 11 12 13  14  15  16
        8, 0, 4, 24, 8, 16, 24, 8,
    //  17 18  19  20 21 22 23
        16, 8, 16, 8, 16, 8, 16
    };


    constexpr uint MAX_DEPTH = 8;
    constexpr uint ACTIVE_THREADS = 0xFFFFFF;

    const uint cur_joint = threadIdx.x;
    const uint cur_object_in_batch = blockDim.y * blockIdx.x + threadIdx.y;
    const uint cur_object_id = local_q_ids[cur_object_in_batch];

    if ((cur_joint >= NUM_JOINTS) || (cur_object_in_batch >= batch_size)) {
        return;
    }

    const uint full_offset = cur_object_id * NUM_JOINTS + cur_joint;
    const uint batch_offset = cur_object_in_batch * NUM_JOINTS + cur_joint;

    int next_joint = PARENTS[cur_joint];
    const float3 cur_joint_pos = joints[full_offset];
    const float3 parent_joint_pos = shuffle_float3(
        ACTIVE_THREADS, cur_joint_pos, max(next_joint, 0));
    const float3 rel_joint =
        (cur_joint == 0) ? cur_joint_pos : cur_joint_pos - parent_joint_pos;

    const quatf cur_q_raw{local_q[full_offset]};
    const se3f cur_trf(cur_q_raw.std(), rel_joint);
    const se3f a0_inv{quatf{a0_inv_q[full_offset]}, a0_inv_t[full_offset]};

    se3f output_result(quatf{output_q[batch_offset]}, output_t[batch_offset]);
    float4 grad_out_q = grad_output_q[batch_offset];
    float3 grad_out_t = grad_output_t[batch_offset];

    // grad: output_result = result.compose(a0_inv);
    se3f result = output_result.compose(a0_inv.inverse());
    auto [
        grad_result_q, grad_result_t,
        grad_a0_inv_q, grad_a0_inv_t
    ] = se3f::compose_backward(result, a0_inv, grad_out_q, grad_out_t);

    d_a0_inv_q[full_offset] = grad_a0_inv_q;
    d_a0_inv_t[full_offset] = grad_a0_inv_t;

    // grad: result.t = result.t - result.q.apply(cur_joint_pos)
    auto [
        grad_result_q2, grad_cur_joint_pos
    ] = quatf::apply_backward(result.q, cur_joint_pos, -grad_result_t);
    grad_result_q += grad_result_q2;
    result.t += result.q.apply(cur_joint_pos);

    constexpr int num_warps = 32;
    __shared__ float4 grad_local_q[num_warps][NUM_JOINTS];
    // __shared__ float3 grad_local_t[num_warps][NUM_JOINTS];

    uint warp_id = threadIdx.y;

    grad_local_q[warp_id][cur_joint] = {0.0f, 0.0f, 0.0f, 0.0f};
    // grad_local_t[warp_id][cur_joint] = {0.0f, 0.0f, 0.0f};
    __syncwarp();

    int cur_path_joint = 0;
    int init_path_idx = joint_path_indices[cur_joint];

    #pragma unroll
    for (uint i = 0; i < MAX_DEPTH+1; ++i) {
        se3f a(
            shuffle_quat(ACTIVE_THREADS, cur_trf.q, cur_path_joint),
            shuffle_float3(ACTIVE_THREADS, cur_trf.t, cur_path_joint)
        );
        if (cur_path_joint != cur_joint) {
            se3f b = a.inverse().compose(result);
            // printf("joint=%d i=%d| cur_path_joint=%d | "
            //        "a.q = [%.4f, %.4f, %.4f, %.4f] a.t=[%.4f, %.4f, %.4f], "
            //        "b.q = [%.4f, %.4f, %.4f, %.4f] b.t=[%.4f, %.4f, %.4f],"
            //        "grad.q=[%.4f, %.4f, %.4f, %.4f] grad.t=[%.4f, %.4f, %.4f]\n",
            //     cur_joint, i, cur_path_joint,
            //     a.q.q.x, a.q.q.y, a.q.q.z, a.q.q.w,
            //     a.t.x, a.t.y, a.t.z,
            //     b.q.q.x, b.q.q.y, b.q.q.z, b.q.q.w,
            //     b.t.x, b.t.y, b.t.z,
            //     grad_result_q.x, grad_result_q.y, grad_result_q.z, grad_result_q.w,
            //     grad_result_t.x, grad_result_t.y, grad_result_t.z);
            const auto [
                da_q, da_t, db_q, db_t
            ] = se3f::compose_backward(a, b, grad_result_q, grad_result_t);

            atomicAdd_float4(&grad_local_q[warp_id][cur_path_joint], da_q);
            // atomicAdd_float3(grad_local_t[warp_id][cur_path_joint], da_t);

            result = b;
            grad_result_q = db_q;
            grad_result_t = db_t;
            // advance joint
            cur_path_joint = root_to_leaf_paths[init_path_idx + i];
        }
        __syncwarp();
    }

    grad_local_q[warp_id][cur_joint] += grad_result_q;
    // grad_local_t[warp_id][cur_joint] += grad_result_t;
    __syncwarp();

    d_local_q[full_offset] = quatf::std_backward(
        cur_q_raw, grad_local_q[warp_id][cur_joint]);
    // d_rel_joint = grad_local_t[warp_id][cur_joint];
    // repurpose grad_local_q to store d_cur_joint,
    // scatter gradients via parent indices
}


std::tuple<at::Tensor, at::Tensor, at::Tensor> smpl_local_to_global_bwd(
    const at::Tensor& local_q_ids,    // [B,]
    const at::Tensor& local_q,        // [N, 24, 4]
    const at::Tensor& joints,         // [N, 24, 3]
    const at::Tensor& a0_inv_q,       // [N, 24, 4]
    const at::Tensor& a0_inv_t,       // [N, 24, 3]
    const at::Tensor& output_q,
    const at::Tensor& output_t,
    const at::Tensor& grad_output_q,  // [B, 24, 4]
    const at::Tensor& grad_output_t   // [B, 24, 3]
) {
    const uint batch_size = local_q_ids.numel();

    const at::cuda::OptionalCUDAGuard device_guard(local_q.device());

    at::Tensor d_local_q = at::zeros_like(local_q);
    at::Tensor d_a0_inv_q = at::zeros_like(a0_inv_q);
    at::Tensor d_a0_inv_t = at::zeros_like(a0_inv_t);

    constexpr uint num_objects_per_block = 16;
    dim3 threads(32, num_objects_per_block, 1);
    uint num_blocks =
        (batch_size + num_objects_per_block - 1) / num_objects_per_block;

    smpl_local_to_global_bwd_kernel
        <<<num_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        batch_size,
        (uint*)local_q_ids.data_ptr(),
        (float4*)local_q.data_ptr(),
        (float3*)joints.data_ptr(),
        (float4*)a0_inv_q.data_ptr(),
        (float3*)a0_inv_t.data_ptr(),
        (float4*)output_q.data_ptr(),
        (float3*)output_t.data_ptr(),
        (float4*)grad_output_q.data_ptr(),
        (float3*)grad_output_t.data_ptr(),
        (float4*)d_local_q.data_ptr(),
        (float4*)d_a0_inv_q.data_ptr(),
        (float3*)d_a0_inv_t.data_ptr()
    );

    return {d_local_q, d_a0_inv_q, d_a0_inv_t};
}


void bind_smpl_local_to_global(py::module_& m) {
    m.def("smpl_local_to_global_fwd", &smpl_local_to_global_fwd,
        py::arg("local_q_ids"), py::arg("local_q"),
        py::arg("joints"), py::arg("a0_inv_q"), py::arg("a0_inv_t")
         );

    m.def("smpl_local_to_global_bwd", &smpl_local_to_global_bwd,
        py::arg("local_q_ids"), py::arg("local_q"),
        py::arg("joints"), py::arg("a0_inv_q"), py::arg("a0_inv_t"),
        py::arg("output_q"), py::arg("output_t"),
        py::arg("grad_output_q"), py::arg("grad_output_t"));
}