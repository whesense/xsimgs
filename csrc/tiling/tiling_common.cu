#include "tiling_common.cuh"
#include "bind_tiling.h"

#include <toast/utils/vector_math.h>
#include <toast/utils/launch_helpers.h>
#include <toast/utils/tensor_helpers.h>

#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <c10/cuda/CUDAGuard.h>


#define CUB_WRAPPER(func, ...)                                               \
    do {                                                                     \
        size_t temp_storage_bytes = 0;                                       \
        func(nullptr, temp_storage_bytes, __VA_ARGS__);                      \
        auto &caching_allocator = *::c10::cuda::CUDACachingAllocator::get(); \
        auto temp_storage = caching_allocator.allocate(temp_storage_bytes);  \
        func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);           \
    } while (false)

namespace cg = cooperative_groups;


__global__ void isect_offset_encode_kernel(
    const uint32_t n_isects,
    const uint32_t n_tiles,
    const uint32_t tile_n_bits,

    const int64_t *__restrict__ isect_ids,
    int32_t *__restrict__ offsets
) {
    // e.g., ids: [1, 1, 1, 3, 3], n_tiles = 6
    // counts: [0, 3, 0, 2, 0, 0]
    // cumsum: [0, 3, 3, 5, 5, 5]
    // offsets: [0, 0, 3, 3, 5, 5]
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= n_isects)
        return;

    int64_t isect_id_curr = isect_ids[idx] >> 32;
    int64_t cid_curr = isect_id_curr >> tile_n_bits;
    int64_t tid_curr = isect_id_curr & ((1 << tile_n_bits) - 1);
    int64_t id_curr = cid_curr * n_tiles + tid_curr;

    if (idx == 0) {
        // write out the offsets until the first valid tile (inclusive)
        for (uint32_t i = 0; i < id_curr + 1; ++i)
            offsets[i] = static_cast<int32_t>(idx);
    }
    if (idx == n_isects - 1) {
        // write out the rest of the offsets
        for (uint32_t i = id_curr + 1; i < n_tiles; ++i)
            offsets[i] = static_cast<int32_t>(n_isects);
    }

    if (idx > 0) {
        // visit the current and previous isect_id and check if the (cid, tile_id)
        // pair changes.
        int64_t isect_id_prev = isect_ids[idx - 1] >> 32; // shift out the depth
        if (isect_id_prev == isect_id_curr)
            return;

        // write out the offsets between the previous and current tiles
        int64_t cid_prev = isect_id_prev >> tile_n_bits;
        int64_t tid_prev = isect_id_prev & ((1 << tile_n_bits) - 1);
        int64_t id_prev = cid_prev * n_tiles + tid_prev;
        for (uint32_t i = id_prev + 1; i < id_curr + 1; ++i)
            offsets[i] = static_cast<int32_t>(idx);
    }
}

at::Tensor isect_offset_encode(
    const at::Tensor &isect_ids, // [n_isects]
    const uint32_t tile_width,
    const uint32_t tile_height
) {
    TORCH_CHECK(isect_ids.is_cuda() && isect_ids.is_contiguous(),
        "isect_ids must be contiguous cuda tensor")

    uint32_t n_isects = isect_ids.size(0);
    auto offsets = at::empty(
        {tile_height, tile_width},
              isect_ids.options().dtype(at::kInt));

    if (!n_isects) {
        return offsets.fill_(0);
    }

    uint32_t n_tiles = tile_width * tile_height;
    uint32_t tile_n_bits = (uint32_t)floorf(log2f(float(n_tiles))) + 1;

    launch_kernel_cuda(
        n_isects, isect_ids.device(), isect_offset_encode_kernel,
        n_isects, n_tiles, tile_n_bits,
        isect_ids.data_ptr<int64_t>(), offsets.data_ptr<int32_t>());

    return offsets;
}


std::tuple<at::Tensor, at::Tensor, at::Tensor> sort_intersections(
    const at::Tensor &tiles_per_gauss,
    const at::Tensor &isect_ids,
    const at::Tensor &flatten_ids,
    const uint tile_n_bits,
    bool sort,
    bool double_buffer
) {
    const uint n_isects = isect_ids.numel();
    auto stream = at::cuda::getCurrentCUDAStream();
    const at::cuda::OptionalCUDAGuard device_guard(isect_ids.device());


    // optionally sort the Gaussians by isect_ids
    if (n_isects && sort) {
        at::Tensor isect_ids_sorted = at::empty_like(isect_ids);
        at::Tensor flatten_ids_sorted = at::empty_like(flatten_ids);

        // https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
        // DoubleBuffer reduce the auxiliary memory usage from O(N+P) to O(P)
        if (double_buffer) {
            // Create a set of DoubleBuffers to wrap pairs of device pointers
            cub::DoubleBuffer<int64_t> d_keys(
                isect_ids.data_ptr<int64_t>(),
                isect_ids_sorted.data_ptr<int64_t>());
            cub::DoubleBuffer<int32_t> d_values(
                flatten_ids.data_ptr<int32_t>(),
                flatten_ids_sorted.data_ptr<int32_t>());
            CUB_WRAPPER(cub::DeviceRadixSort::SortPairs, d_keys, d_values, n_isects, 0,
                        32 + tile_n_bits, stream);
            switch (d_keys.selector) {
                case 0: // sorted items are stored in isect_ids
                    isect_ids_sorted = isect_ids;
                    break;
                case 1: // sorted items are stored in isect_ids_sorted
                    break;
            }
            switch (d_values.selector) {
                case 0: // sorted items are stored in flatten_ids
                    flatten_ids_sorted = flatten_ids;
                    break;
                case 1: // sorted items are stored in flatten_ids_sorted
                    break;
            }
            // printf("DoubleBuffer d_keys selector: %d\n", d_keys.selector);
            // printf("DoubleBuffer d_values selector: %d\n", d_values.selector);
        } else {
            CUB_WRAPPER(cub::DeviceRadixSort::SortPairs, isect_ids.data_ptr<int64_t>(),
                        isect_ids_sorted.data_ptr<int64_t>(),
                        flatten_ids.data_ptr<int32_t>(),
                        flatten_ids_sorted.data_ptr<int32_t>(), n_isects, 0,
                        32 + tile_n_bits, stream);
        }
        return std::make_tuple(
            tiles_per_gauss,
            isect_ids_sorted,
            flatten_ids_sorted);
    }

    return std::make_tuple(tiles_per_gauss, isect_ids, flatten_ids);
}


void bind_common_tiling(py::module_& m) {
    m.def("isect_offset_encode", &isect_offset_encode,
        py::arg("isect_ids"), py::arg("tile_width"), py::arg("tile_height"));
}