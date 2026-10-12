#ifndef XSIMGS_TILING_COMMON_CUH
#define XSIMGS_TILING_COMMON_CUH

#include <toast/utils/vector_math.h>
#include <ATen/core/Tensor.h>


std::tuple<at::Tensor, at::Tensor, at::Tensor>
sort_intersections(
    const at::Tensor& tiles_per_gauss,
    const at::Tensor& isect_ids,
    const at::Tensor& flatten_ids,
    uint tile_n_bits,
    bool sort,
    bool double_buffer
);

at::Tensor isect_offset_encode(
    const at::Tensor &isect_ids, // [n_isects]
    uint32_t tile_width,
    uint32_t tile_height
);


HOSTDEVICE void fill_tiles(
    const int idx,
    const float depth,

    const int2& tile_min,
    const int2& tile_max,
    const uint32_t n_tiles_azim,

    int64_t *__restrict__ isect_ids,
    int32_t *__restrict__ flatten_ids,
    uint32_t& offset
) {
    auto depth_id_enc = static_cast<int64_t>(*(int32_t *) &(depth));

    for (int32_t i = tile_min.y; i <= tile_max.y; ++i) {
        for (int32_t j = tile_min.x; j <= tile_max.x; ++j) {
            int64_t tile_id = i * n_tiles_azim + j;
            // tile id (32 bits) | depth (32 bits)
            isect_ids[offset] = (tile_id << 32) | depth_id_enc;
            flatten_ids[offset] = idx;
            ++offset;
        }
    }
}

HOSTDEVICE void fill_tiles_exclusive(
    const int idx,
    const float depth,

    const int2& tile_min,
    const int2& tile_max,
    const uint32_t n_tiles_azim,

    int64_t *__restrict__ isect_ids,
    int32_t *__restrict__ flatten_ids,
    uint32_t& offset
) {
    auto depth_id_enc = static_cast<int64_t>(*(int32_t *) &(depth));

    for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
        for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
            int64_t tile_id = i * n_tiles_azim + j;
            // tile id (32 bits) | depth (32 bits)
            isect_ids[offset] = (tile_id << 32) | depth_id_enc;
            flatten_ids[offset] = idx;
            ++offset;
        }
    }
}


#endif //XSIMGS_TILING_COMMON_CUH
