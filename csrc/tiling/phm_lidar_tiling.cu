#include "tiling_common.cuh"
#include "bind_tiling.h"
#include "../cameras/spherical.h"
#include <cameras/ut_kernels.h>

#include <toast/utils/vector_math.h>
#include <toast/utils/launch_helpers.h>
#include <toast/utils/tensor_helpers.h>
#include <cooperative_groups.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/cumsum.h>

namespace cg = cooperative_groups;


HOSTDEVICE void find_bounds_decreasing(
    const float vmin,
    const float vmax,
    const uint N,
    const float* values,
    int& tile_min,
    int& tile_max
) {
    int i = 0;
    while ((i < N) && (values[i] >= vmax)) {
        ++i;
    }
    tile_min = min(max(i - 1, 0), N - 1);
    while ((i < N) && (values[i] >= vmin)) {
        ++i;
    }
    tile_max = min(max(i - 1, 0), N - 1);
}

HOSTDEVICE void find_bounds_increasing(
    const float vmin,
    const float vmax,
    const uint N,
    const float* values,
    int& tile_min,
    int& tile_max
) {
    int i = 0;
    while ((i < N) && (values[i] < vmin)) {++i;}
    tile_min = min(max(i - 1, 0), N - 1);

    while ((i < N) && (values[i] < vmax)) {++i;}
    tile_max = min(max(i - 1, 0), N - 1);
}


template <bool limits_decreasing = true>
HOSTDEVICE void tile_limits_irregular(
    const float2& cur_mean,
    const float2& cur_radii,
    const float u_min,
    const float u_max,
    const float u_per_tile,
    const int n_tiles_elev,
    const float* elev_boundaries,

    int2& tile_min,
    int2& tile_max,
    int& num_tiles
) {
    tile_min = {0, 0}; tile_max = {-1, -1};

    const float point_azim_low = fmaxf(cur_mean.x - cur_radii.x, u_min);
    const float point_azim_high = fminf(cur_mean.x + cur_radii.x, u_max);
    if (point_azim_high <= point_azim_low) {
        return;
    }

    tile_min.x = static_cast<int32_t>(floorf(point_azim_low / u_per_tile));
    tile_max.x = static_cast<int32_t>(ceilf(point_azim_high / u_per_tile)) - 1;

    if constexpr (limits_decreasing) {
        find_bounds_decreasing(
            cur_mean.y - cur_radii.y, cur_mean.y + cur_radii.y,
            n_tiles_elev, elev_boundaries,
            tile_min.y, tile_max.y
        );
    } else {
        find_bounds_increasing(
            cur_mean.y - cur_radii.y, cur_mean.y + cur_radii.y,
            n_tiles_elev, elev_boundaries,
            tile_min.y, tile_max.y);
    }

    num_tiles += (tile_max.y - tile_min.y + 1) * (tile_max.x - tile_min.x + 1);
}




template <bool limits_decreasing = false>
__global__ void isect_lidar_tiles_kernel(
    const uint32_t N,
    const float u_per_tile,
    const int n_tiles_azim,
    const int n_tiles_elev,
    const float4 ut_params,
    const float2 img_size,

    // for second projection
    const LidarCamera* __restrict__ camera,
    const float3* __restrict__ point,
    const float3* __restrict__ point_velocity,
    const float3* __restrict__ scale,
    const float4* __restrict__ quat,
    const float* __restrict__ opacity,

    // data
    const char *__restrict__ flags,
    const float2 *__restrict__ means2d, // [N, 2]
    const float2 *__restrict__ radii,   // [N, 2]
    const float *__restrict__ depths,   // [N]
    const int64_t *__restrict__ cum_tiles_per_gauss, // [N]

    const float *__restrict__ elev_boundaries,      // [M]

    int32_t *__restrict__ tiles_per_gauss, // [C, N] or [nnz]
    int64_t *__restrict__ isect_ids,       // [n_isects]
    int32_t *__restrict__ flatten_ids      // [n_isects]
) {
    const uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N)
        return;

    const bool first_pass = cum_tiles_per_gauss == nullptr;
    float2 cur_radii = radii[idx];
    const char flag = flags[idx];

    if (cur_radii.x <= 0 || cur_radii.y <= 0 || flag == 0) {
        if (first_pass)
            tiles_per_gauss[idx] = 0;
        return;
    }

    int2 tile_min, tile_max; int num_tiles = 0;
    uint32_t offset = 0;

    // const float u_pose = (camera->pose_time - camera->trigger_time) / camera->rolling_time.x;
    // float pose_azimuth = camera->impl.phi_min + u_pose * camera->impl.phi_range;
    //
    // float min_azimuth = camera->impl.phi_min;
    // float max_azimuth = camera->impl.phi_min + camera->impl.phi_range;
    float u_min = 0.0f, u_max = 1.0f;
    const float u_pose =
        fabsf(camera->rolling_time.x) > 0.0f ?
            (camera->pose_time - camera->trigger_time) / camera->rolling_time.x : 0.5f;

    if (flag == 2) {
        if (camera->rolling_time.x > 0) {
            u_max = u_pose;
        } else {
            u_min = u_pose;
        }
    }

    // const float2 px_to_angle = float2{camera->impl.phi_range, camera->impl.theta_range} / img_size;
    // const float2 vmin{camera->impl.phi_min, camera->impl.theta_min};
    // const float2 mean2d = means2d[idx] * px_to_angle + vmin;
    // cur_radii *= float2{fabsf(px_to_angle.x), fabsf(px_to_angle.y)};

    float2 mean2d = means2d[idx] / img_size;
    cur_radii /= img_size;
    float depth;

    tile_limits_irregular<limits_decreasing>(mean2d, cur_radii, u_min, u_max, u_per_tile,
        n_tiles_elev, elev_boundaries,
        tile_min, tile_max, num_tiles);

    if (!first_pass) {
        if (idx > 0) {
            offset = cum_tiles_per_gauss[idx - 1];
        }
        depth = depths[idx];
        fill_tiles(idx, depth, tile_min, tile_max,
            n_tiles_azim, isect_ids, flatten_ids, offset);
    }

    if (flag == 2) {
        // Step 1: project. Get cur_mean, cur_radii, depth
        float tmp;
        float3 deltas[3];

        const float3 p_world = point[idx];
        const float3 p_velocity = point_velocity ?
            point_velocity[idx] : float3{0.0f, 0.0f, 0.0f};
        const float3 p_scale = scale[idx];
        const quatf p_quat{quat[idx]};
        const float p_opacity = opacity ? opacity[idx] : 1.0f;

        unscented_transform_deltas(ut_params.w, p_quat, p_scale, deltas);

        constexpr float residual_thr = 1e-4f;
        constexpr float max_residual = 0.002f;
        constexpr float max_margin = 1.0f;

        bool result = unscented_transform_projection<5, false, LidarCamera>(
           ProjectionMode::PositiveDelta,
           camera, ut_params, img_size, residual_thr, max_residual, max_margin,
           p_world, p_velocity, p_opacity, deltas,
           mean2d, cur_radii, depth, tmp
        );
        // Step 2: tile_limits, update counts
        if (result) {
            if (camera->rolling_time.x > 0) {
                // advance to the next tile
                u_min = ceilf(u_pose / u_per_tile) * u_per_tile + 1e-6f;
                u_max = 1.0f;
            } else {
                u_min = 0.0f;
                // advance to the previous tile
                u_max = floorf(u_pose / u_per_tile) * u_per_tile - 1e-6f;
            }
            tile_limits_irregular<limits_decreasing>(mean2d / img_size, cur_radii / img_size,
                u_min, u_max, u_per_tile,
                n_tiles_elev, elev_boundaries,
                tile_min, tile_max, num_tiles);

            // Step 3: fill values
            if (!first_pass) {
                fill_tiles(idx, depth, tile_min, tile_max,
                    n_tiles_azim, isect_ids, flatten_ids, offset);
            }
        }
    }

    if (first_pass) {
        tiles_per_gauss[idx] = num_tiles;
    }
}




std::tuple<at::Tensor, at::Tensor, at::Tensor> isect_lidar_tiles(
    const at::Tensor& camera_pack,
    const at::Tensor& points,
    const std::optional<at::Tensor>& points_velocity,
    const at::Tensor& scale,
    const at::Tensor& quat,
    const std::optional<at::Tensor>& opacity,

    const at::Tensor& flag,
    const at::Tensor& means2d,
    const at::Tensor& radii,
    const at::Tensor& depths,
    const at::Tensor& elev_boundaries,

    int tile_size_x,
    float img_width,
    float img_height,
    float ut_alpha,
    float ut_beta,
    float ut_kappa,

    const bool sort,
    const bool double_buffer
) {
    TORCH_CHECK(camera_pack.is_cuda(), "camera_pack must be CUDA tensor")
    check_same_device(
        camera_pack,
        points, points_velocity.value_or(points), scale, quat,
        flag, means2d, radii, depths, elev_boundaries);
    all_contiguous(
        camera_pack,
        points, points_velocity.value_or(points), scale, quat,
        flag, means2d, radii, depths, elev_boundaries);

    uint32_t N = means2d.numel() / 2;

    const int n_tiles_azim = (static_cast<int>(img_width) + tile_size_x - 1) / tile_size_x;
    const int n_tiles_elev = static_cast<int>(elev_boundaries.size(0));
    const uint32_t n_tiles = n_tiles_azim * n_tiles_elev;

    const float u_per_tile = static_cast<float>(tile_size_x) / img_width;
    const float v_per_tile = ceilf(img_height / static_cast<float>(n_tiles_elev)) / img_height;

    const uint32_t tile_n_bits = static_cast<uint32_t>(
        floor(log2(static_cast<float>(n_tiles)))) + 1;
    assert(tile_n_bits <= 32);

    // first pass: compute number of tiles per gaussian
    at::Tensor tiles_per_gauss =
        at::zeros_like(depths, depths.options().dtype(at::kInt));

    const float2 img_size{img_width, img_height};
    const float4 ut_params = ut_weights(ut_alpha, ut_beta, ut_kappa);

    launch_kernel_cuda<CUDA_DEFAULT_NUM_THREADS/2>(
        N, means2d.device(), isect_lidar_tiles_kernel<false>,

        N, u_per_tile, n_tiles_azim, n_tiles_elev, ut_params, img_size,
        (LidarCamera*)camera_pack.data_ptr(),
        (float3*)points.data_ptr(),
        points_velocity.has_value() ? (float3*)points_velocity->data_ptr() : nullptr,
        (float3*)scale.data_ptr(),
        (float4*)quat.data_ptr(),
        opacity.has_value() ? (float*)opacity->data_ptr() : nullptr,
        (char*)flag.data_ptr(),
        (float2*)means2d.data_ptr(),
        (float2*)radii.data_ptr(),
        depths.data_ptr<float>(),
        nullptr,
        elev_boundaries.data_ptr<float>(),

        tiles_per_gauss.data_ptr<int32_t>(),
        nullptr, nullptr
        );

    const auto cum_tiles_per_gauss =
        at::cumsum(tiles_per_gauss.view({-1}), 0);
    const int64_t n_isects = cum_tiles_per_gauss[-1].item<int64_t>();

    // second pass: compute isect_ids and flatten_ids as a packed tensor
    const auto isect_ids =
        at::empty({n_isects}, depths.options().dtype(at::kLong));
    const auto flatten_ids =
        at::empty({n_isects}, depths.options().dtype(at::kInt));

    if (!n_isects) {
        return {tiles_per_gauss, isect_ids, flatten_ids};
    }

    launch_kernel_cuda<CUDA_DEFAULT_NUM_THREADS/2>(
        N, means2d.device(), isect_lidar_tiles_kernel<false>,

        N, u_per_tile, n_tiles_azim, n_tiles_elev, ut_params, img_size,
        (LidarCamera*)camera_pack.data_ptr(),
        (float3*)points.data_ptr(),
        points_velocity.has_value() ? (float3*)points_velocity->data_ptr() : nullptr,
        (float3*)scale.data_ptr(),
        (float4*)quat.data_ptr(),
        opacity.has_value() ? (float*)opacity->data_ptr() : nullptr,
        (char*)flag.data_ptr(),
        (float2*)means2d.data_ptr(),
        (float2*)radii.data_ptr(),
        depths.data_ptr<float>(),
        (int64_t*)cum_tiles_per_gauss.data_ptr(),
        elev_boundaries.data_ptr<float>(),

        nullptr, (int64_t*)isect_ids.data_ptr(), (int32_t*)flatten_ids.data_ptr()
        );

    return sort_intersections(
        tiles_per_gauss, isect_ids, flatten_ids,
        tile_n_bits, sort, double_buffer);
}


void bind_lidar_tiling(py::module_& m) {
    m.def("isect_lidar_tiles", &isect_lidar_tiles,
        py::arg("camera_pack"),
        py::arg("points"),
        py::arg("points_velocity"),
        py::arg("scale"),
        py::arg("quat"),
        py::arg("opacity"),
        py::arg("flag"),
        py::arg("means2d"),
        py::arg("radii"),
        py::arg("depths"),
        py::arg("elev_boundaries"),
        py::arg("tile_size_x"),
        py::arg("img_width"),
        py::arg("img_height"),
        py::arg("ut_alpha") = DEFAULT_UT_ALPHA,
        py::arg("ut_beta") = DEFAULT_UT_BETA,
        py::arg("ut_kappa") = DEFAULT_UT_KAPPA,
        py::arg("sort") = true,
        py::arg("double_buffer") = true);
}