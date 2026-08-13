// Copyright 2025 Nerfstudio Team (original gsplat code)
// Modifications Copyright 2026 the xsimgs authors
// SPDX-License-Identifier: Apache-2.0
//
// Derived from gsplat (https://github.com/nerfstudio-project/gsplat),
// licensed under the Apache License 2.0.
// This file has been substantially modified from the original.

#include "eval3d_bind.h"

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <glm/glm.hpp>

#include <toast/utils/vector_math.h>
#include <toast/utils/launch_helpers.h>
#include <toast/utils/tensor_helpers.h>

#include "common.h"
#include "utils.h"

#include "tile_shared_memory.h"


#define BWD_SHMEM_LAYOUT(X) \
    X(int32_t, particle_ids, 1) \
    X(glm::vec4, xyz_opac, 1) \
    X(glm::vec3, scales, 1) \
    X(glm::vec4, quats, 1) \
    X(float, rgbs, CDIM)

template <uint32_t CDIM>
SHARED_MEM_LAYOUT(TileSharedMemoryBwd, BWD_SHMEM_LAYOUT)


#define MAX_ALPHA 0.99f
#define MAX_KERNEL_DENSITY_CUTOFF 0.0113f


template <int CDIM>
HOSTDEVICE float grad_alpha(
    const float* particle_rgb,
    const float* buffer,
    const float T_final,
    const float T,
    const float ra,
    const float* v_render_c,
    const float v_render_a,
    const float v_render_depth,
    const float depth_buffer,
    const float t_hit
) {
    float v_alpha = 0.0f;
    #pragma unroll
    for (uint32_t k = 0; k < CDIM; ++k) {
        v_alpha += (particle_rgb[k] * T - buffer[k] * ra) * v_render_c[k];
    }
    v_alpha += (t_hit * T - depth_buffer * ra) * v_render_depth;
    v_alpha += T_final * ra * v_render_a;
    return v_alpha;
}


template <uint32_t CDIM>
__global__ void rasterize_eval3d_bwd_kernel(
    const IndexingData data,
    const vec3 *__restrict__ means,
    const vec4 *__restrict__ quats,
    const vec3 *__restrict__ scales,
    const float *__restrict__ colors,
    const float *__restrict__ opacities,

    const vec3* __restrict__ ray_origs,
    const vec3* __restrict__ ray_dirs,

    const int32_t *__restrict__ tile_offsets,
    const int32_t *__restrict__ flatten_ids,

    const float *__restrict__ render_alphas,
    const int32_t *__restrict__ last_ids,
    // grad outputs
    const float *__restrict__ v_render_colors,
    const float *__restrict__ v_render_depth,
    const float *__restrict__ v_render_alphas,

    // grad inputs
    vec3 *__restrict__ v_means,
    vec4 *__restrict__ v_quats,
    vec3 *__restrict__ v_scales,
    float *__restrict__ v_colors,
    float *__restrict__ v_opacities,
    vec3 *__restrict__ v_ray_origs,
    vec3 *__restrict__ v_ray_dirs
) {
    auto block = cg::this_thread_block();
    const uint32_t block_size = block.size();
    const Indices indices = data.get_indices(block);

    const vec3 ray_d = indices.inside ? ray_dirs[indices.pix_id] : vec3(0.0);
    const vec3 ray_o = indices.inside ? ray_origs[indices.pix_id] : vec3(0.0);

    bool done = indices.inside;
    IntersectionRange range(data, indices, tile_offsets);

    extern __shared__ char shared_mem[];
    TileSharedMemoryBwd<CDIM> sh_mem(shared_mem, block_size);

    float T_final = 1.0f - render_alphas[indices.pix_id];
    float T = T_final;
    constexpr int COLOR_SIZE = (CDIM > 0) ? CDIM: 1;
    float buffer[COLOR_SIZE] = {};
    const int32_t bin_final = done ? last_ids[indices.pix_id] : 0;

    float v_render_c[COLOR_SIZE];
    #pragma unroll
    for (uint32_t k = 0; k < CDIM; ++k) {
        v_render_c[k] = indices.inside ? v_render_colors[indices.pix_id * CDIM + k] : 0.0f;
    }
    const float v_render_d = indices.inside ? v_render_depth[indices.pix_id] : 0.0f;
    const float v_render_a = indices.inside ? v_render_alphas[indices.pix_id] : 0.0f;

    vec3 v_ray_o = {0.f, 0.f, 0.f};
    vec3 v_ray_d = {0.f, 0.f, 0.f};
    float depth_buffer = 0.0f;
    float last_hit_T = 0.0f;

    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final =
        cg::reduce(warp, bin_final, cg::greater<int>());
    for (uint32_t b = 0; b < range.num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        const int32_t batch_end = range.range_end - 1 - block_size * b;
        const int32_t batch_size = min(block_size, batch_end + 1 - range.range_start);
        const int32_t idx = batch_end - tr;
        if (idx >= range.range_start) {
            int32_t isect_id = flatten_ids[idx]; // flatten index in [B * C * N] or [nnz]
            sh_mem.particle_ids[tr] = isect_id;
            const vec3 xyz = means[isect_id];
            sh_mem.xyz_opac[tr] = {xyz.x, xyz.y, xyz.z, opacities[isect_id]};
            sh_mem.scales[tr] = scales[isect_id];
            sh_mem.quats[tr] = quats[isect_id];
            #pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                sh_mem.rgbs[tr * CDIM + k] = colors[isect_id * CDIM + k];
            }
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();

        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
            bool valid = done;
            if (batch_end - t > bin_final) {
                valid = false;
            }
            float alpha;
            float opac;
            float vis;

            mat3 R, S;
            vec3 xyz;
            vec3 scale;
            vec4 quat;
            mat3 Mt;
            vec3 o_minus_mu, gro, grd, grd_n, gcrod;
            float grayDist, power;

            if (valid) {
                const vec4 xyz_opac = sh_mem.xyz_opac[t];
                opac = xyz_opac[3];
                xyz = {xyz_opac[0], xyz_opac[1], xyz_opac[2]};
                scale = sh_mem.scales[t];
                quat = sh_mem.quats[t];

                R = quat_to_rotmat(quat);
                S = mat3(
                    1.0f / scale[0],
                    0.f,
                    0.f,
                    0.f,
                    1.0f / scale[1],
                    0.f,
                    0.f,
                    0.f,
                    1.0f / scale[2]
                );
                Mt = glm::transpose(R * S);
                o_minus_mu = ray_o - xyz;
                gro = Mt * o_minus_mu;
                grd = Mt * ray_d;

                grd_n = safe_normalize(grd);
                gcrod = glm::cross(grd_n, gro);
                grayDist = glm::dot(gcrod, gcrod);
                power = -0.5f * grayDist;

                vis = __expf(power);
                alpha = min(MAX_ALPHA, opac * vis);
                if (alpha < 1.f / 255.f) {  //  || vis <= MAX_KERNEL_DENSITY_CUTOFF
                    valid = false;
                }
                last_hit_T = -dot(grd, gro) / dot(grd, grd);
            }

            // if all threads are inactive in this warp, skip this loop
            if (!warp.any(valid)) {
                continue;
            }
            float v_rgb_local[COLOR_SIZE] = {};
            vec3 v_mean_local = {0.f, 0.f, 0.f};
            vec3 v_scale_local = {0.f, 0.f, 0.f};
            vec4 v_quat_local = {0.f, 0.f, 0.f, 0.f};
            float v_opacity_local = 0.f;

            if (valid) {
                // compute the current T for this gaussian
                float ra = 1.0f / fmaxf(1e-6f, 1.0f - alpha);
                T *= ra;
                // update v_rgb for this gaussian
                const float fac = alpha * T;
                set_from<CDIM>(v_rgb_local, v_render_c, fac);

                // contribution from this pixel
                float v_alpha = grad_alpha<CDIM>(
                    sh_mem.rgbs + t * CDIM,
                    buffer,
                    T_final, T, ra, v_render_c, v_render_a,
                    v_render_d, depth_buffer, last_hit_T
                );

                // Add contribution from hit distance (if enabled)
                const float dd = dot(grd, grd);
                const float od = dot(gro, grd);

                const float dL_dhitT = v_render_d * fac;
                vec3 v_gro = -grd * dL_dhitT / dd;
                vec3 v_grd = (2.0f * od * grd - dd * gro) * dL_dhitT / (dd * dd);

                if (opac * vis <= MAX_ALPHA) {
                    const float v_vis = opac * v_alpha;
                    float v_gradDist = -0.5f * vis * v_vis;
                    vec3 v_gcrod = 2.0f * v_gradDist * gcrod;
                    vec3 v_grd_n = -glm::cross(v_gcrod, gro);
                    v_gro += glm::cross(v_gcrod, grd_n);
                    v_grd += safe_normalize_bwd(grd, v_grd_n);
                    mat3 v_Mt = glm::outerProduct(v_grd, ray_d) +
                        glm::outerProduct(v_gro, o_minus_mu);
                    mat3 trMt = glm::transpose(Mt);
                    vec3 v_o_minus_mu = trMt * v_gro;

                    v_mean_local += -v_o_minus_mu;

                    // Compute ray gradients
                    // From o_minus_mu = ray_o - xyz, we get:
                    v_ray_o += v_o_minus_mu;
                    // From grd = Mt * ray_d, we get:
                    v_ray_d += trMt * v_grd;

                    quat_scale_to_preci_half_vjp(
                        quat, scale, R, glm::transpose(v_Mt), v_quat_local, v_scale_local
                    );
                    v_opacity_local = vis * v_alpha;
                }

                blend_from<CDIM>(buffer, sh_mem.rgbs + t * CDIM, fac);
                depth_buffer += fac * last_hit_T;
            }

            warpSum<CDIM>(v_rgb_local, warp);
            warpSum(v_mean_local, warp);
            warpSum(v_scale_local, warp);
            warpSum(v_quat_local, warp);
            warpSum(v_opacity_local, warp);

            if (warp.thread_rank() == 0) {
                const int particle_idx = sh_mem.particle_ids[t];
                atomicAdd<CDIM>(v_colors + CDIM * particle_idx, v_rgb_local);
                atomicAdd((float*)v_means + 3 * particle_idx, v_mean_local);
                atomicAdd((float*)v_scales + 3 * particle_idx, v_scale_local);
                atomicAdd((float*)v_quats + 4 * particle_idx, v_quat_local);
                gpuAtomicAdd(v_opacities + particle_idx, v_opacity_local);
            }
        }
    }

    if (indices.inside) {
        v_ray_origs[indices.pix_id] = v_ray_o;
        v_ray_dirs[indices.pix_id] = v_ray_d;
    }
}


std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    at::optional<torch::Tensor>,
    torch::Tensor,
    torch::Tensor
> rasterize_eval3d_bwd(
    const torch::Tensor& points,
    const torch::Tensor& scale,
    const torch::Tensor& quat,
    const torch::Tensor& opacity,
    const at::optional<torch::Tensor>& colors,

    const torch::Tensor& ray_origins,
    const torch::Tensor& ray_directions,
    const uint32_t tile_size_x,
    const uint32_t tile_size_y,

    const torch::Tensor& tile_offsets,
    const torch::Tensor& flatten_ids,

    const torch::Tensor& render_alpha,
    const torch::Tensor& last_ids,

    const at::optional<torch::Tensor>& v_render_colors,
    const torch::Tensor& v_render_depth,
    const torch::Tensor& v_render_alpha
) {
    if (v_render_colors.has_value() && !v_render_colors->is_contiguous()) {
        throw std::runtime_error("v_render_colors must be contiguous");
    }
    all_contiguous(v_render_depth, v_render_alpha);

    const at::cuda::OptionalCUDAGuard device_guard(points.device());

    const uint width = ray_origins.size(1);
    const uint height = ray_origins.size(0);

    IndexingData indexing_data{
        .N = static_cast<uint32_t>(points.numel() / 3),
        .n_isects = static_cast<uint32_t>(flatten_ids.numel()),
        .image_width = width,
        .image_height = height,
        .tile_width = tile_size_x,
        .tile_height = tile_size_y,
        .num_tiles_width = static_cast<uint32_t>(tile_offsets.size(1)),
        .num_tiles_height = static_cast<uint32_t>(tile_offsets.size(0))
    };

    torch::Tensor v_points = torch::zeros_like(points);
    torch::Tensor v_scale = torch::zeros_like(scale);
    torch::Tensor v_quat = torch::zeros_like(quat);
    torch::Tensor v_opacity = torch::zeros_like(opacity);
    at::optional<torch::Tensor> v_colors;
    uint32_t cdim = 0;
    if (colors.has_value()) {
        v_colors = torch::zeros_like(colors.value());
        cdim = colors->size(1);
    }
    torch::Tensor v_ray_o = torch::zeros_like(ray_origins);
    torch::Tensor v_ray_d = torch::zeros_like(ray_directions);

    dim3 threads = {indexing_data.tile_width, indexing_data.tile_height, 1}; \
    dim3 grid = {1, indexing_data.num_tiles_height, indexing_data.num_tiles_width}; \

#define launch_kernel_case(CDIM)\
    case CDIM: { \
        const auto shmem_size = TileSharedMemoryBwd<CDIM>::mem_size(indexing_data.tile_width, indexing_data.tile_height); \
        set_max_shared_mem<TileSharedMemoryBwd<CDIM>>(\
            reinterpret_cast<const void*>(&rasterize_eval3d_bwd_kernel<CDIM>),\
            indexing_data.tile_width, indexing_data.tile_height); \
        rasterize_eval3d_bwd_kernel<CDIM><<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(\
            indexing_data,\
            (vec3*)points.data_ptr(),\
            (vec4*)quat.data_ptr(),\
            (vec3*)scale.data_ptr(),\
            colors.has_value() ? (float*)colors->data_ptr() : nullptr,\
            (float*)opacity.data_ptr(),\
            (vec3*)ray_origins.data_ptr(),\
            (vec3*)ray_directions.data_ptr(),\
            (int32_t*)tile_offsets.data_ptr(),\
            (int32_t*)flatten_ids.data_ptr(),\
            (float*)render_alpha.data_ptr(),\
            (int32_t*)last_ids.data_ptr(),\
            v_render_colors.has_value() ? (float*)v_render_colors->data_ptr() : nullptr,\
            (float*)v_render_depth.data_ptr(),\
            (float*)v_render_alpha.data_ptr(),\
            (vec3*)v_points.data_ptr(),\
            (vec4*)v_quat.data_ptr(),\
            (vec3*)v_scale.data_ptr(),\
            v_colors.has_value() ? (float*)v_colors->data_ptr() : nullptr,\
            (float*)v_opacity.data_ptr(),\
            (vec3*)v_ray_o.data_ptr(),\
            (vec3*)v_ray_d.data_ptr()\
        ); \
        break;\
    }

    switch (cdim) {
        ndim_cases(launch_kernel_case)
        default: throw std::runtime_error("unsupported feature dimension: " + std::to_string(cdim));
    }

    return {v_points, v_quat,  v_scale, v_opacity, v_colors, v_ray_o, v_ray_d};
}


void bind_rasterization_eval3d_bwd(py::module &m) {
    m.def("rasterize_eval3d_bwd", &rasterize_eval3d_bwd);
}