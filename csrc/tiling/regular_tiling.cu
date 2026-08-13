#include "tiling_common.cuh"
#include "bind_tiling.h"
#include <toast/utils/vector_math.h>
#include <toast/utils/launch_helpers.h>
#include <toast/utils/tensor_helpers.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;



template <bool first_pass>
__global__ void isect_tiles_kernel(
    const uint32_t N,
    const float2 tile_size,
    const int2 n_tiles,

    // data
    const float2 *__restrict__ means2d, // [N, 2]
    const float2 *__restrict__ radii,   // [N, 2]
    const float *__restrict__ depths,   // [N]

    const int32_t *__restrict__ cum_tiles_per_gauss, // [N]

    int32_t *__restrict__ tiles_per_gauss, // [C, N] or [nnz]
    int64_t *__restrict__ isect_ids,       // [n_isects]
    int32_t *__restrict__ flatten_ids      // [n_isects]
) {
    const uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N)
        return;

    float2 cur_radii = radii[idx];
    if (cur_radii.x <= 0 || cur_radii.y <= 0) { // || flags[idx] == 0
        if (first_pass)
            tiles_per_gauss[idx] = 0;
        return;
    }

    const float2 mean2d = means2d[idx] / tile_size;
    cur_radii /= tile_size;

    const int2 tile_min = {
        min(n_tiles.x, max(0,
            static_cast<int>(floorf(mean2d.x - cur_radii.x)))),
        min(n_tiles.y, max(0,
            static_cast<int>(floorf(mean2d.y - cur_radii.y))))
    };
    const int2 tile_max = {
        min(n_tiles.x, max(0,
            static_cast<int>(ceilf(mean2d.x + cur_radii.x)))),
        min(n_tiles.y, max(0,
            static_cast<int>(ceilf(mean2d.y + cur_radii.y))))
    };
    const int num_tiles = (tile_max.y - tile_min.y) * (tile_max.x - tile_min.x);

    if ((!first_pass) && (num_tiles > 0)) {
        uint offset = 0;
        if (idx > 0) {
            offset = cum_tiles_per_gauss[idx - 1];
        }
        fill_tiles_exclusive(idx, depths[idx], tile_min, tile_max,
            n_tiles.x, isect_ids, flatten_ids, offset);
    } else {
        tiles_per_gauss[idx] = num_tiles;
    }

}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> isect_regular_tiles(
    const torch::Tensor& means2d,
    const torch::Tensor& radii,
    const torch::Tensor& depths,
    int img_width,
    int img_height,
    int tile_size_x,
    int tile_size_y,
    bool sort,
    bool double_buffer
) {
    check_same_device(means2d, radii, depths);
    all_contiguous(means2d, radii, depths);
    const uint32_t N = means2d.numel() / 2;

    const int2 n_tiles{
        (img_width + tile_size_x - 1) / tile_size_x,
        (img_height + tile_size_y - 1) / tile_size_y
    };
    const float2 tile_size{
        static_cast<float>(tile_size_x),
        static_cast<float>(tile_size_y)
    };

    const uint32_t tile_n_bits = static_cast<uint32_t>(
        floorf(log2f(static_cast<float>(n_tiles.x * n_tiles.y)))) + 1;
    assert(tile_n_bits <= 32);

    auto tiles_per_gauss =
        torch::empty_like(depths, depths.options().dtype(torch::kInt32));

    launch_kernel_cuda<256>(N, means2d.device(), isect_tiles_kernel<true>,
        N, tile_size, n_tiles,
        (float2*)means2d.data_ptr(),
        (float2*)radii.data_ptr(),
        (float*)depths.data_ptr(),
        nullptr, (int32_t*)tiles_per_gauss.data_ptr(), nullptr, nullptr);

    const auto cum_tiles_per_gauss =
        torch::cumsum(tiles_per_gauss.view({-1}), 0, torch::kInt32);
    const auto n_isects = cum_tiles_per_gauss[-1].item<int32_t>();

    const auto isect_ids =
        torch::empty({n_isects}, depths.options().dtype(torch::kInt64));
    const auto flatten_ids =
        torch::empty({n_isects}, depths.options().dtype(torch::kInt32));

    if (!n_isects) {
        return {tiles_per_gauss, isect_ids, flatten_ids};
    }

    launch_kernel_cuda<256>(N, means2d.device(), isect_tiles_kernel<false>,
        N, tile_size, n_tiles,
        (float2*)means2d.data_ptr(),
        (float2*)radii.data_ptr(), (float*)depths.data_ptr(),
        (int32_t*)cum_tiles_per_gauss.data_ptr(),
        (int32_t*)tiles_per_gauss.data_ptr(),
        (int64_t*)isect_ids.data_ptr(),
        (int32_t*)flatten_ids.data_ptr());

    return sort_intersections(
        tiles_per_gauss, isect_ids, flatten_ids,
        tile_n_bits, sort, double_buffer);
}


void bind_regular_tiling(py::module_& m) {
    m.def("isect_regular_tiles", &isect_regular_tiles,
        py::arg("means2d"),
        py::arg("radii"),
        py::arg("depth"),
        py::arg("img_width"), py::arg("img_height"),
        py::arg("tile_size_x"), py::arg("tile_size_y"),
        py::arg("sort") = true, py::arg("double_buffer") = true);
}