#include "smpl.h"

#include <toast/math/se3.h>
#include <toast/utils/launch_helpers.h>
#include <toast/utils/tensor_helpers.h>




__global__ void smpl_lbs_skinning_kernel(
    const uint num_objects,
    const uint num_vertices,
    const uint* __restrict__ valid_ids,
    const float4* __restrict__ global_q,  // [B, 24, 4]
    const float3* __restrict__ global_t,  // []
    const float3* __restrict__ vertices, // [N, 6890, 3]
    const float4* __restrict__ quats, // [N, 6890, 4]
    const float* __restrict__ weights, // [N, 24, 6890]
    const uint* __restrict__ instance_ids, // [N, ]
    const float3* __restrict__ instance_t, // [K, 3]
    const float3* __restrict__ instance_v, // [K, 3]

    const uint global_offset,
    float3* __restrict__ out_vertices,
    float4* __restrict__ out_quats,
    float3* __restrict__ out_velocities,
    char* __restrict__ out_mask
) {
    const uint global_vertex_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (global_vertex_id >= num_vertices * num_objects) {
        return;
    }
    const uint vertex_in_object_id = global_vertex_id % num_vertices;
    const uint object_id = global_vertex_id / num_vertices;
    const uint real_object_id = valid_ids[object_id];
    const uint trf_offset = object_id * NUM_JOINTS;
    uint weights_offset =
        NUM_JOINTS * num_vertices * real_object_id + vertex_in_object_id;

    float4 weighted_trf[3] = {0.0f, 0.0f, 0.0f};

    #pragma unroll
    for (uint i = 0; i < NUM_JOINTS; ++i) {
        const float cur_weight = weights[weights_offset];
        se3f cur_se3{quatf{global_q[trf_offset + i]}, global_t[trf_offset + i]};
        float4 cur_mat[3];
        cur_se3.to_matrix_3x4(cur_mat);
        weighted_trf[0] += cur_weight * cur_mat[0];
        weighted_trf[1] += cur_weight * cur_mat[1];
        weighted_trf[2] += cur_weight * cur_mat[2];
        weights_offset += num_vertices;
    }

    const uint full_vertex_id = real_object_id * num_vertices + vertex_in_object_id;

    const uint src_instance_id = instance_ids[real_object_id];
    const float3 this_instance_t = instance_t[src_instance_id];

    float3 vert = vertices[full_vertex_id];
    float4 vert_homo = {vert.x, vert.y, vert.z, 1.0f};

    const uint output_idx = global_offset + full_vertex_id;

    out_vertices[output_idx] = float3{
        dot(vert_homo, weighted_trf[0]),
        dot(vert_homo, weighted_trf[1]),
        dot(vert_homo, weighted_trf[2]),
    } + this_instance_t;

    // TODO: need some cheap re-orthogonalization of matrix.
    // TODO: may be use Newton-Shulz iteration: R = 0.5 * (R + R^{-1}^T)
    // TODO: or just decompose this matrix into scaling and rotation, and
    // TODO: scale gaussians accordingly?
    quatf lbs_quat = quatf::from_matrix(weighted_trf).std();
    out_quats[output_idx] = lbs_quat.quat_mul(quatf{quats[full_vertex_id]}).q;

    if (out_velocities) {
        out_velocities[output_idx] = instance_v[src_instance_id];
    }

    if (out_mask) {
        out_mask[output_idx] = true;
    }
}


void smpl_lbs_skinning_fwd(
    const torch::Tensor& valid_ids, // [B,]
    const torch::Tensor& global_q, // [B, 24, 4]
    const torch::Tensor& global_t, // [B, 24, 3]
    const torch::Tensor& vertices, // [N, 6890, 3]
    const torch::Tensor& quats,   // [N, 6890, 4]
    const torch::Tensor& weights, // [N, 24, 6890]
    const torch::Tensor& instance_ids,  // [N,]
    const torch::Tensor& instance_t, // [K, 3]
    const torch::Tensor& instance_v, // [K, 3]
    const uint global_offset,
    const torch::Tensor& out_positions,
    const torch::Tensor& out_rotations,
    const std::optional<torch::Tensor>& out_velocities,
    const std::optional<torch::Tensor>& out_mask
) {
    const at::cuda::OptionalCUDAGuard device_guard(instance_ids.device());

    const uint num_objects = valid_ids.numel();
    const uint num_vertices = vertices.size(1);

    constexpr uint num_threads = 1024;
    const uint num_blocks = (num_vertices * num_objects + num_threads - 1) / num_threads;

    smpl_lbs_skinning_kernel<<<num_blocks, num_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_objects, num_vertices,
        (uint*)valid_ids.data_ptr(),
        (float4*)global_q.data_ptr(),
        (float3*)global_t.data_ptr(),
        (float3*)vertices.data_ptr(),
        (float4*)quats.data_ptr(),
        (float*)weights.data_ptr(),
        (uint*) instance_ids.data_ptr(),
        (float3*)instance_t.data_ptr(),
        (float3*)instance_v.data_ptr(),

        global_offset,
        (float3*)out_positions.data_ptr(),
        (float4*)out_rotations.data_ptr(),
        out_velocities.has_value() ? (float3*)out_velocities->data_ptr() : nullptr,
        out_mask.has_value() ? (char*)out_mask->data_ptr() : nullptr
    );
}


__global__ void smpl_lbs_skinning_bwd_kernel(
    const uint num_objects,
    const uint num_vertices,
    const uint* __restrict__ valid_ids,
    const uint* __restrict__ instance_ids, // [N,]
    const float4* __restrict__ global_q,   // [B, 24, 4]
    const float3* __restrict__ global_t,   // [B, 24, 3]
    const float3* __restrict__ vertices,   // [N, V, 3]
    const float4* __restrict__ quats,      // [N, V, 4]
    const float* __restrict__ weights,     // [N, 24, V]

    const uint global_offset,
    const float3* __restrict__ grad_out_vertices,
    const float4* __restrict__ grad_out_quats,
    const float3* __restrict__ grad_out_velocities,

    float4* __restrict__ d_global_q,    // [B, 24, 4]
    float3* __restrict__ d_global_t,    // [B, 24, 3]
    float3* __restrict__ d_vertices,    // [N, V, 3]
    float4* __restrict__ d_quats,       // [N, V, 4]
    float* __restrict__ d_weights,      // [N, 24, V]
    float3* __restrict__ d_instance_v,  // [K, 3]
    float3* __restrict__ d_instance_t   // [K, 3]
) {
    const uint global_vertex_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (global_vertex_id >= num_vertices * num_objects) return;

    const uint vertex_in_object_id = global_vertex_id % num_vertices;
    const uint object_id            = global_vertex_id / num_vertices;
    const uint real_object_id       = valid_ids[object_id];
    const uint trf_offset           = object_id * NUM_JOINTS;
    const uint full_vertex_id       = real_object_id * num_vertices + vertex_in_object_id;
    const uint src_instance_id      = instance_ids[real_object_id];
    const uint output_idx           = global_offset + full_vertex_id;

    uint weights_offset = NUM_JOINTS * num_vertices * real_object_id + vertex_in_object_id;
    float4 weighted_trf[3] = {
        {0.f,0.f,0.f,0.f},
        {0.f,0.f,0.f,0.f},
        {0.f,0.f,0.f,0.f}
    };

    #pragma unroll
    for (uint i = 0; i < NUM_JOINTS; ++i) {
        const float w = weights[weights_offset];
        se3f cur_se3{quatf{global_q[trf_offset + i]}, global_t[trf_offset + i]};
        float4 cur_mat[3];
        cur_se3.to_matrix_3x4(cur_mat);
        weighted_trf[0] += w * cur_mat[0];
        weighted_trf[1] += w * cur_mat[1];
        weighted_trf[2] += w * cur_mat[2];
        weights_offset += num_vertices;
    }

    const float3 vert      = vertices[full_vertex_id];
    const quatf from_mat_out = quatf::from_matrix(weighted_trf);
    const quatf lbs_quat     = from_mat_out.std();
    const quatf vert_quat{quats[full_vertex_id]};

    const float3 grad_v   = grad_out_vertices[output_idx];
    const float4 grad_q   = grad_out_quats[output_idx];

    float4 d_lbs_q_raw, d_vert_q;
    quatf::quat_mul_backward(lbs_quat, vert_quat, grad_q,
        d_lbs_q_raw, d_vert_q);

    const float4 d_from_mat_out = quatf::std_backward(from_mat_out, d_lbs_q_raw);

    float4 d_wtrf_from_q[3] = {
        {0.f,0.f,0.f,0.f},
        {0.f,0.f,0.f,0.f},
        {0.f,0.f,0.f,0.f}
    };
    quatf::from_matrix_backward(weighted_trf, d_from_mat_out, d_wtrf_from_q);

    const float4 d_wtrf_from_v[3] = {
        {vert.x*grad_v.x, vert.y*grad_v.x, vert.z*grad_v.x, grad_v.x},
        {vert.x*grad_v.y, vert.y*grad_v.y, vert.z*grad_v.y, grad_v.y},
        {vert.x*grad_v.z, vert.y*grad_v.z, vert.z*grad_v.z, grad_v.z},
    };
    const float3 d_vert = {
        grad_v.x*weighted_trf[0].x + grad_v.y*weighted_trf[1].x + grad_v.z*weighted_trf[2].x,
        grad_v.x*weighted_trf[0].y + grad_v.y*weighted_trf[1].y + grad_v.z*weighted_trf[2].y,
        grad_v.x*weighted_trf[0].z + grad_v.y*weighted_trf[1].z + grad_v.z*weighted_trf[2].z,
    };

    const float4 d_wtrf[3] = {
        d_wtrf_from_q[0] + d_wtrf_from_v[0],
        d_wtrf_from_q[1] + d_wtrf_from_v[1],
        d_wtrf_from_q[2] + d_wtrf_from_v[2],
    };

    weights_offset = NUM_JOINTS * num_vertices * real_object_id + vertex_in_object_id;

    #pragma unroll
    for (uint i = 0; i < NUM_JOINTS; ++i) {
        const float w = weights[weights_offset];
        se3f cur_se3{quatf{global_q[trf_offset + i]}, global_t[trf_offset + i]};
        float4 cur_mat[3];
        cur_se3.to_matrix_3x4(cur_mat);

        d_weights[weights_offset] =
            dot(cur_mat[0], d_wtrf[0]) +
                dot(cur_mat[1], d_wtrf[1]) +
                    dot(cur_mat[2], d_wtrf[2]);

        const float4 d_cur_mat[3] = {w * d_wtrf[0], w * d_wtrf[1], w * d_wtrf[2]};
        const auto [d_q_j, d_t_j] =
            se3f::to_matrix_3x4_backward(cur_se3, d_cur_mat);

        atomicAdd(&d_global_q[trf_offset + i].x, d_q_j.x);
        atomicAdd(&d_global_q[trf_offset + i].y, d_q_j.y);
        atomicAdd(&d_global_q[trf_offset + i].z, d_q_j.z);
        atomicAdd(&d_global_q[trf_offset + i].w, d_q_j.w);

        atomicAdd(&d_global_t[trf_offset + i].x, d_t_j.x);
        atomicAdd(&d_global_t[trf_offset + i].y, d_t_j.y);
        atomicAdd(&d_global_t[trf_offset + i].z, d_t_j.z);

        weights_offset += num_vertices;
    }

    d_vertices[full_vertex_id] = d_vert;
    d_quats[full_vertex_id] = d_vert_q;

    atomicAdd(&d_instance_t[src_instance_id].x, grad_v.x);
    atomicAdd(&d_instance_t[src_instance_id].y, grad_v.y);
    atomicAdd(&d_instance_t[src_instance_id].z, grad_v.z);

    if (grad_out_velocities && d_instance_v) {
        const float3 gv = grad_out_velocities[output_idx];
        atomicAdd(&d_instance_v[src_instance_id].x, gv.x);
        atomicAdd(&d_instance_v[src_instance_id].y, gv.y);
        atomicAdd(&d_instance_v[src_instance_id].z, gv.z);
    }
}


std::tuple<
    torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor>
smpl_lbs_skinning_bwd(
    const torch::Tensor& valid_ids,       // [B,]
    const torch::Tensor& instance_ids,    // [N,]
    const torch::Tensor& global_q,        // [B, 24, 4]
    const torch::Tensor& global_t,        // [B, 24, 3]
    const torch::Tensor& vertices,        // [N, V, 3]
    const torch::Tensor& quats,           // [N, V, 4]
    const torch::Tensor& weights,         // [N, 24, V]
    const uint global_offset,
    const uint num_instances,             // K
    const torch::Tensor& grad_out_vertices,
    const torch::Tensor& grad_out_quats,
    const std::optional<torch::Tensor>& grad_out_velocities
) {
    const at::cuda::OptionalCUDAGuard device_guard(instance_ids.device());

    const uint num_objects  = valid_ids.numel();
    const uint num_vertices = vertices.size(1);

    torch::Tensor d_global_q   = torch::zeros_like(global_q);
    torch::Tensor d_global_t   = torch::zeros_like(global_t);
    torch::Tensor d_vertices   = torch::zeros_like(vertices);
    torch::Tensor d_quats      = torch::zeros_like(quats);
    torch::Tensor d_weights    = torch::zeros_like(weights);
    torch::Tensor d_instance_v = torch::zeros({num_instances, 3}, global_q.options());
    torch::Tensor d_instance_t = torch::zeros({num_instances, 3}, global_q.options());

    constexpr uint num_threads = 256;
    const uint num_blocks = (num_vertices * num_objects + num_threads - 1) / num_threads;

    smpl_lbs_skinning_bwd_kernel<<<num_blocks, num_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_objects, num_vertices,
        (uint*)valid_ids.data_ptr(),
        (uint*)instance_ids.data_ptr(),
        (float4*)global_q.data_ptr(),
        (float3*)global_t.data_ptr(),
        (float3*)vertices.data_ptr(),
        (float4*)quats.data_ptr(),
        (float*)weights.data_ptr(),
        global_offset,
        (float3*)grad_out_vertices.data_ptr(),
        (float4*)grad_out_quats.data_ptr(),
        grad_out_velocities.has_value() ? (float3*)grad_out_velocities->data_ptr() : nullptr,
        (float4*)d_global_q.data_ptr(),
        (float3*)d_global_t.data_ptr(),
        (float3*)d_vertices.data_ptr(),
        (float4*)d_quats.data_ptr(),
        (float*)d_weights.data_ptr(),
        grad_out_velocities.has_value() ? (float3*)d_instance_v.data_ptr() : nullptr,
        (float3*)d_instance_t.data_ptr()
    );

    return {
        d_global_q, d_global_t,
        d_vertices, d_quats, d_weights,
        d_instance_v, d_instance_t
    };
}


void bind_smpl_lbs_skinning(py::module_& m) {
    m.def("smpl_lbs_skinning", &smpl_lbs_skinning_fwd,
        py::arg("valid_ids"), py::arg("global_q"), py::arg("global_t"),
        py::arg("vertices"), py::arg("quats"), py::arg("weights"),
        py::arg("instance_ids"), py::arg("instance_t"), py::arg("instance_v"),
        py::arg("global_offset"),
        py::arg("out_positions"),
        py::arg("out_rotations"),
        py::arg("out_velocities"),
        py::arg("out_mask"));

    m.def("smpl_lbs_skinning_bwd", &smpl_lbs_skinning_bwd,
        py::arg("valid_ids"), py::arg("instance_ids"),
        py::arg("global_q"), py::arg("global_t"),
        py::arg("vertices"), py::arg("quats"), py::arg("weights"),
        py::arg("global_offset"), py::arg("num_instances"),
        py::arg("grad_out_vertices"), py::arg("grad_out_quats"),
        py::arg("grad_out_velocities") = py::none());
}