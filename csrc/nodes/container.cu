#include "container.h"

#include <ATen/cuda/Atomic.cuh>
#include <toast/utils/launch_helpers.h>
#include <toast/utils/tensor_helpers.h>
#include <toast/math/se3.h>

#include "utils.cuh"
#include <ATen/ops/zeros_like.h>


template <bool with_offsets>
__global__ void copy_gs_rigid_transform_fwd_kernel(
    const uint num_particles,
    const uint global_offset,

    const float3* __restrict__ positions,  // [N, 3]
    const float4* __restrict__ rotations,  // [N, 4]
    const uint* __restrict__ instance_ids, // [N, ]

    const float4* __restrict__ inst_q,     // [K, 4]
    const float3* __restrict__ inst_t,     // [K, 3]
    const float3* __restrict__ inst_v,     // [K, 3]
    const float3* __restrict__ inst_w,     // [K, 3]
    const char* __restrict__ inst_mask,    // [K,]

    const uint* __restrict__ per_inst_offsets, // [K,]
    const uint* __restrict__ gs_inst_offsets,  // [N, ]

    float3* __restrict__ out_positions,  // [B, 3]
    float4* __restrict__ out_rotations,  // [B, 4]
    float3* __restrict__ out_velocities, // [B, 3]
    char* __restrict__ out_mask          // [B,]
) {
    const uint particle_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (particle_idx >= num_particles) {
        return;
    }

    const uint particle_instance_id = instance_ids[particle_idx];
    if (!inst_mask[particle_instance_id]) {
        return;
    }

    const quatf cur_inst_q{inst_q[particle_instance_id]};
    const float3 cur_inst_t = inst_t[particle_instance_id];
    const float3 cur_inst_v = inst_v[particle_instance_id];
    const float3 cur_inst_w = inst_w[particle_instance_id];

    const float3 cur_pos = positions[particle_idx];
    const quatf cur_q{rotations[particle_idx]};

    uint result_idx;
    if (!with_offsets) {
        result_idx = global_offset + particle_idx;
    } else {
        result_idx = global_offset +
                per_inst_offsets[particle_instance_id] +
                gs_inst_offsets[particle_idx];
    }

    const float3 p_rot = cur_inst_q.apply(cur_pos);
    out_positions[result_idx] = p_rot + cur_inst_t;
    out_rotations[result_idx] = cur_inst_q.quat_mul(cur_q).q;
    if (out_velocities) {
        out_velocities[result_idx] = cur_inst_v + cross(cur_inst_w, p_rot);
    }
    if (out_mask) {
        out_mask[result_idx] = true;
    }
}


template <bool with_offsets>
__global__ void copy_gs_rigid_transform_bwd_kernel(
    const uint num_particles,
    const uint global_offset,

    const float3* __restrict__ positions,  // [N, 3]
    const float4* __restrict__ rotations,  // [N, 4]
    const uint* __restrict__ instance_ids, // [N, ]

    const float4* __restrict__ inst_q,     // [K, 4]
    const float3* __restrict__ inst_w,     // [K, 3]
    const char* __restrict__ inst_mask,    // [K,]

    const uint* __restrict__ per_inst_offsets, // [K,]
    const uint* __restrict__ gs_inst_offsets,  // [N, ]

    const float3* __restrict__ v_positions,  // [B, 3]
    const float4* __restrict__ v_rotations,  // [B, 4]
    const float3* __restrict__ v_velocities, // [B, 3]

    float3* __restrict__ grad_positions,
    float4* __restrict__ grad_rotations,

    float4* __restrict__ grad_inst_q,
    float3* __restrict__ grad_inst_t,
    float3* __restrict__ grad_inst_v,
    float3* __restrict__ grad_inst_w
) {
    const uint particle_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (particle_idx >= num_particles) {
        return;
    }

    const uint particle_instance_id = instance_ids[particle_idx];
    if (!inst_mask[particle_instance_id]) {
        return;
    }

    const quatf cur_inst_q{inst_q[particle_instance_id]};
    const float3 cur_inst_w = inst_w[particle_instance_id];

    const float3 cur_pos = positions[particle_idx];
    const quatf cur_q{rotations[particle_idx]};

    uint result_idx;
    if (!with_offsets) {
        result_idx = global_offset + particle_idx;
    } else {
        result_idx = global_offset +
                per_inst_offsets[particle_instance_id] +
                gs_inst_offsets[particle_idx];
    }


    // out_positions[result_idx] = cur_inst_q.apply(cur_pos) + cur_inst_t;
    const float3 v_pos_out = v_positions[result_idx];
    atomicAdd_float3(grad_inst_t + particle_instance_id, v_pos_out);

    float3 v_p_rot = v_pos_out;

    if (v_velocities) {
        //     out_velocities[result_idx] =
        //         cur_inst_v + cur_inst_q.apply(cross(cur_inst_w, cur_pos));
        const float3 v_vel_out = v_velocities[result_idx];
        atomicAdd_float3(grad_inst_v + particle_instance_id, v_vel_out);

        const float3 p_rot = cur_inst_q.apply(cur_pos);
        atomicAdd_float3(grad_inst_w + particle_instance_id, cross(p_rot, v_vel_out));
        v_p_rot += cross(v_vel_out, cur_inst_w);
    }

    auto [v_inst_q, v_pos] = quatf::apply_backward(cur_inst_q, cur_pos, v_p_rot);
    grad_positions[particle_idx] = v_pos;

    // out_rotations[result_idx] = cur_inst_q.quat_mul(cur_q).q;
    float4 v_inst_q2, v_quat;
    quatf::quat_mul_backward(cur_inst_q, cur_q, v_rotations[result_idx], v_inst_q2, v_quat);
    v_inst_q += v_inst_q2;
    grad_rotations[particle_idx] = v_quat;

    atomicAdd_float4(grad_inst_q + particle_instance_id, v_inst_q);
}

#define bool_template_launch(param, N, device, fn, ...) \
    if (param) { \
        launch_kernel_cuda(N, device, fn<true>, __VA_ARGS__); \
    } else { \
        launch_kernel_cuda(N, device, fn<false>, __VA_ARGS__); \
    }


void copy_gs_rigid_transform_fwd(
    const at::Tensor& positions,
    const at::Tensor& rotations,
    const at::Tensor& instance_id,

    const at::Tensor& inst_q,
    const at::Tensor& inst_t,
    const at::Tensor& inst_v,
    const at::Tensor& inst_w,
    const at::Tensor& inst_mask,

    const at::Tensor& out_positions,
    const at::Tensor& out_rotations,
    const std::optional<at::Tensor>& out_velocities,
    const std::optional<at::Tensor>& out_mask,

    uint buffer_offset,
    const std::optional<at::Tensor>& per_instance_offsets,
    const std::optional<at::Tensor>& gs_inst_offsets
) {
    uint num_particles = positions.size(0);
    bool with_offsets = gs_inst_offsets.has_value();

    bool_template_launch(with_offsets, num_particles, positions.device(), copy_gs_rigid_transform_fwd_kernel,
        num_particles,
        buffer_offset,
        (float3*)positions.data_ptr(),
        (float4*)rotations.data_ptr(),
        (uint*)instance_id.data_ptr(),

        (float4*)inst_q.data_ptr(),
        (float3*)inst_t.data_ptr(),
        (float3*)inst_v.data_ptr(),
        (float3*)inst_w.data_ptr(),
        (char*)inst_mask.data_ptr(),

        per_instance_offsets.has_value() ? (uint*)per_instance_offsets->data_ptr(): nullptr,
        gs_inst_offsets.has_value() ? (uint*)gs_inst_offsets->data_ptr(): nullptr,

        (float3*)out_positions.data_ptr(),
        (float4*)out_rotations.data_ptr(),
        out_velocities.has_value() ? (float3*)out_velocities->data_ptr() : nullptr,
        out_mask.has_value() ? (char*)out_mask->data_ptr() : nullptr)
}


auto copy_gs_rigid_transform_bwd(
    const at::Tensor& positions,
    const at::Tensor& rotations,
    const at::Tensor& instance_id,

    const at::Tensor& inst_q,
    const at::Tensor& inst_w,
    const at::Tensor& inst_mask,

    uint buffer_offset,
    const std::optional<at::Tensor>& per_instance_offsets,
    const std::optional<at::Tensor>& gs_inst_offsets,

    const at::Tensor& v_positions,
    const at::Tensor& v_rotations,
    const std::optional<at::Tensor>& v_velocities
) {
    uint num_particles = positions.size(0);
    bool with_offsets = gs_inst_offsets.has_value();

    auto grad_positions = at::zeros_like(positions);
    auto grad_rotations = at::zeros_like(rotations);

    auto grad_q = at::zeros_like(inst_q);
    auto grad_t = at::zeros_like(inst_w);
    auto grad_v = at::zeros_like(inst_w);
    auto grad_w = at::zeros_like(inst_w);

    bool_template_launch(with_offsets, num_particles, positions.device(), copy_gs_rigid_transform_bwd_kernel,
        num_particles,
        buffer_offset,
        (float3*)positions.data_ptr(),
        (float4*)rotations.data_ptr(),
        (uint*)instance_id.data_ptr(),

        (float4*)inst_q.data_ptr(),
        (float3*)inst_w.data_ptr(),
        (char*)inst_mask.data_ptr(),

        per_instance_offsets.has_value() ? (uint*)per_instance_offsets->data_ptr(): nullptr,
        gs_inst_offsets.has_value() ? (uint*)gs_inst_offsets->data_ptr(): nullptr,

        (float3*)v_positions.data_ptr(),
        (float4*)v_rotations.data_ptr(),
        v_velocities.has_value() ? (float3*)v_velocities->data_ptr() : nullptr,

        (float3*)grad_positions.data_ptr(),
        (float4*)grad_rotations.data_ptr(),

        (float4*)grad_q.data_ptr(),
        (float3*)grad_t.data_ptr(),
        (float3*)grad_v.data_ptr(),
        (float3*)grad_w.data_ptr())

    return std::tuple{grad_positions, grad_rotations, grad_q, grad_t, grad_v, grad_w};
}


void bind_container_functions(py::module_ &m) {
    m.def("copy_gs_rigid_transform_fwd", &copy_gs_rigid_transform_fwd,
        py::arg("positions"),
        py::arg("rotations"),
        py::arg("instance_id"),
        py::arg("inst_q"),
        py::arg("inst_t"),
        py::arg("inst_v"),
        py::arg("inst_w"),
        py::arg("inst_mask"),
        py::arg("out_positions"),
        py::arg("out_rotations"),
        py::arg("out_velocities"),
        py::arg("out_mask"),
        py::arg("buffer_offset"),
        py::arg("per_instance_offsets") = py::none(),
        py::arg("gs_inst_offsets") = py::none()
    );

    m.def("copy_gs_rigid_transform_bwd", &copy_gs_rigid_transform_bwd,
        py::arg("positions"),
        py::arg("rotations"),
        py::arg("instance_id"),
        py::arg("inst_q"),
        py::arg("inst_w"),
        py::arg("inst_mask"),
        py::arg("buffer_offset"),
        py::arg("per_instance_offsets"),
        py::arg("gs_inst_offsets"),
        py::arg("v_positions"),
        py::arg("v_rotations"),
        py::arg("v_velocities"));
}
