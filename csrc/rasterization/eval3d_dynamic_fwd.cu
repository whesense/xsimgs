// Copyright 2025 Nerfstudio Team (original gsplat code)
// Modifications Copyright 2026 the xsimgs authors
// SPDX-License-Identifier: Apache-2.0
//
// Derived from gsplat (https://github.com/nerfstudio-project/gsplat),
// licensed under the Apache License 2.0.
// This file has been substantially modified from the original.

#include "eval3d_bind.h"

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <glm/glm.hpp>


#include "common.h"
#include "utils.h"

#include "tile_shared_memory.h"


#define FWD_SHMEM_LAYOUT(X) \
    X(int32_t, particle_ids, 1) \
    X(glm::fvec4, xyz_opac, 1) \
    X(glm::fmat3, iscl_rot, 1) \
    X(glm::fvec3, velocity, 1)

SHARED_MEM_LAYOUT(TileSharedMemoryFwd, FWD_SHMEM_LAYOUT)




template <uint32_t CDIM>
__global__ void rasterize_eval3d_dynamic_fwd_kernel(
    const IndexingData data,
    // scene
    const vec3 *__restrict__ means,
    const vec3 *__restrict__ velocity,
    const vec4 *__restrict__ quats,
    const vec3 *__restrict__ scales,
    const float *__restrict__ colors,
    const float *__restrict__ opacities,
    // camera rays
    const vec3* __restrict__ ray_origs,
    const vec3* __restrict__ ray_dirs,
    const float* __restrict__ ray_time_delta,
    // intersections
    const int32_t *__restrict__ tile_offsets,
    const int32_t *__restrict__ flatten_ids,
    // output
    float *__restrict__ render_colors,
    float *__restrict__ render_depth,
    float *__restrict__ render_alphas,
    int32_t *__restrict__ last_ids
) {
    auto block = cg::this_thread_block();
    const uint32_t block_size = block.size();
    const Indices indices = data.get_indices(block);
    bool done = !indices.inside;
    const vec3 ray_d = indices.inside ? ray_dirs[indices.pix_id] : vec3(0.0);
    const vec3 ray_o = indices.inside ? ray_origs[indices.pix_id] : vec3(0.0);
    const float ray_t = (indices.inside && ray_time_delta) ? ray_time_delta[indices.pix_id] : 0.0f;

    IntersectionRange range(data, indices, tile_offsets);
    extern __shared__ char shared_mem[];
    TileSharedMemoryFwd sh_mem(shared_mem, block_size);

    float ray_depth = 0.0f;
    float T = 1.0f;
    uint32_t cur_idx = 0;
    uint32_t tr = block.thread_rank();

    constexpr int COLOR_SIZE = (CDIM > 0) ? CDIM: 1;
    float pix_out[COLOR_SIZE] = {0.f};
    for (uint32_t b = 0; b < range.num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        uint32_t batch_start = range.range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < range.range_end) {
            int32_t isect_id = flatten_ids[idx]; // flatten index in [B * C * N] or [nnz]
            sh_mem.particle_ids[tr] = isect_id;
            const vec3 xyz = means[isect_id];
            const float opac = opacities[isect_id];
            sh_mem.xyz_opac[tr] = {xyz.x, xyz.y, xyz.z, opac};
            const vec4 quat = quats[isect_id];

            mat3 R = quat_to_rotmat(quat);
            const vec3 scale = scales[isect_id];
            // scale_batch[tr] = scale;
            mat3 S = mat3(
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
            mat3 iscl_rot = S * glm::transpose(R);
            sh_mem.iscl_rot[tr] = iscl_rot;
            sh_mem.velocity[tr] = velocity ? velocity[isect_id] : vec3(0.0f);
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        uint32_t batch_size = min(block_size, range.range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const vec4 xyz_opac = sh_mem.xyz_opac[t];
            const float opac = xyz_opac[3];
            const vec3 xyz = vec3(xyz_opac[0], xyz_opac[1], xyz_opac[2]) + ray_t * sh_mem.velocity[t];
            const mat3 iscl_rot = sh_mem.iscl_rot[t];

            const vec3 gro = iscl_rot * (ray_o - xyz);
            const vec3 grd = iscl_rot * ray_d;
            const vec3 gcrod = glm::cross(safe_normalize(grd), gro);
            const float grayDist = glm::dot(gcrod, gcrod);
            const float power = -0.5f * grayDist;

            float alpha = min(0.999f, opac * __expf(power));

            if (alpha < 1.f / 255.f) {
                continue;
            }

            const float hit_t = -dot(gro, grd) / dot(grd, grd);

            const float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) { // this pixel is done: exclusive
                done = true;
                break;
            }

            int32_t isect_id = sh_mem.particle_ids[t];
            const float vis = alpha * T;
            ray_depth += vis * hit_t;

            if constexpr (CDIM > 0) {
                const float *c_ptr = colors + isect_id * CDIM;
                #pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    pix_out[k] += c_ptr[k] * vis;
                }
            }
            cur_idx = batch_start + t;
            T = next_T;
        }
    }

    if (indices.inside) {
        render_alphas[indices.pix_id] = 1.0f - T;
        render_depth[indices.pix_id] = ray_depth;
        #pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            render_colors[indices.pix_id * CDIM + k] = pix_out[k];
        }
        last_ids[indices.pix_id] = static_cast<int32_t>(cur_idx);
    }
}



template <uint32_t CDIM>
void launch_rasterize_eval3d_dynamic_fwd_kernel(
    // Gaussian parameters
    const at::Tensor means,     // [..., N, 3]
    const at::optional<at::Tensor> velocity,
    const at::Tensor quats,     // [..., N, 4]
    const at::Tensor scales,    // [..., N, 3]
    const at::optional<at::Tensor> colors,    // [..., C, N, channels] or [nnz, channels]
    const at::Tensor opacities, // [..., C, N] or [nnz]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size_width,
    const uint32_t tile_size_height,
    // camera
    const at::Tensor ray_origs,
    const at::Tensor ray_dirs,
    const at::optional<at::Tensor> ray_time_delta,
    // intersections
    const at::Tensor tile_offsets, // [..., C, tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // outputs
    at::Tensor renders, // [..., C, image_height, image_width, channels]
    at::Tensor depths,
    at::Tensor alphas,  // [..., C, image_height, image_width]
    at::Tensor last_ids // [..., C, image_height, image_width]
) {
    const at::cuda::OptionalCUDAGuard device_guard(means.device());

    uint32_t N = means.numel() / 3;   // number of gaussians
    uint32_t num_tile_height = tile_offsets.size(-2);
    uint32_t num_tile_width = tile_offsets.size(-1);
    uint32_t n_isects = flatten_ids.numel();

    // Each block covers a tile on the image. In total there are
    // I * tile_height * tile_width blocks.
    dim3 threads = {tile_size_width, tile_size_height, 1};
    dim3 grid = {1, num_tile_height, num_tile_width};

    IndexingData indexing_data{
        .N = N, .n_isects = n_isects,
        .image_width = image_width,
        .image_height = image_height,
        .tile_width = tile_size_width,
        .tile_height = tile_size_height,
        .num_tiles_width = num_tile_width,
        .num_tiles_height = num_tile_height
    };
    set_max_shared_mem<TileSharedMemoryFwd>(
        reinterpret_cast<const void*>(&rasterize_eval3d_dynamic_fwd_kernel<CDIM>),
        indexing_data.tile_width, indexing_data.tile_height
    );
    const uint shmem_size = TileSharedMemoryFwd::mem_size(
        indexing_data.tile_width, indexing_data.tile_height);

    rasterize_eval3d_dynamic_fwd_kernel<CDIM>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            indexing_data,
            reinterpret_cast<vec3 *>(means.data_ptr<float>()),
            velocity.has_value() ? (vec3*)velocity->data_ptr() : nullptr,
            reinterpret_cast<vec4 *>(quats.data_ptr<float>()),
            reinterpret_cast<vec3 *>(scales.data_ptr<float>()),
            colors.has_value() ? colors->data_ptr<float>(): nullptr,
            opacities.data_ptr<float>(),
            // camera rays
            (vec3*)ray_origs.data_ptr(),
            (vec3*)ray_dirs.data_ptr(),
            ray_time_delta.has_value() ? (float*)ray_time_delta->data_ptr() : nullptr,
            // intersections
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            // outputs
            renders.data_ptr<float>(),
            depths.data_ptr<float>(),
            alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>()
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_eval3d_dynamic_fwd(
    // Gaussian parameters
    const at::Tensor means,     // [..., N, 3]
    const at::optional<at::Tensor> velocity,
    const at::Tensor quats,     // [..., N, 4]
    const at::Tensor scales,    // [..., N, 3]
    const at::optional<at::Tensor> colors,    // [..., C, N, channels] or [nnz, channels]
    const at::Tensor opacities, // [..., C, N] or [nnz]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size_width,
    const uint32_t tile_size_height,
    // camera
    const at::Tensor ray_origs,
    const at::Tensor ray_dirs,
    const at::optional<at::Tensor> ray_time_delta,
    // intersections
    const at::Tensor tile_offsets, // [..., C, tile_height, tile_width]
    const at::Tensor flatten_ids   // [n_isects]
) {
    auto opt = means.options();
    uint32_t channels = colors.has_value() ? colors->size(-1) : 0;

    at::Tensor renders = at::empty({image_height, image_width, channels}, opt);
    at::Tensor depths = at::empty({image_height, image_width}, opt);
    at::Tensor alphas = at::empty({image_height, image_width}, opt);
    at::Tensor last_ids = at::empty(
        {image_height, image_width}, opt.dtype(at::kInt));

#define launch_kernel(N)                                          \
    case N:                                                       \
        launch_rasterize_eval3d_dynamic_fwd_kernel<N>(            \
            means,                                                \
            velocity,                                             \
            quats,                                                \
            scales,                                               \
            colors,                                               \
            opacities,                                            \
            image_width,                                          \
            image_height,                                         \
            tile_size_width,                                      \
            tile_size_height,                                     \
            ray_origs,                                            \
            ray_dirs,                                             \
            ray_time_delta,                                       \
            tile_offsets,                                         \
            flatten_ids,                                          \
            renders,                                              \
            depths,                                               \
            alphas,                                               \
            last_ids                                              \
        );                                                        \
        break;

    switch (channels) {
        ndim_cases(launch_kernel)
    default:
        AT_ERROR("Unsupported number of channels: ", channels);
    }
#undef launch_kernel

    return std::make_tuple(renders, depths, alphas, last_ids);
};

void bind_rasterization_eval3d_dynamic_fwd(py::module& m) {
    m.def("rasterize_eval3d_dynamic_fwd", &rasterize_eval3d_dynamic_fwd,
        py::arg("means"), py::arg("velocity"),
        py::arg("quats"), py::arg("scales"), py::arg("colors"),
        py::arg("opacities"), py::arg("image_width"), py::arg("image_height"),
        py::arg("tile_size_width"), py::arg("tile_size_height"),
        py::arg("ray_origs"), py::arg("ray_dirs"), py::arg("ray_time_delta"),
        py::arg("tile_offsets"), py::arg("flatten_ids"));
}