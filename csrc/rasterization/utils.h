#ifndef XSIMGS_RASTERIZAITON_UTILS_H
#define XSIMGS_RASTERIZAITON_UTILS_H

#include <toast/utils/vector_math.h>
#include <cstdint>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <glm/glm.hpp>

#include <ATen/cuda/Atomic.cuh>


namespace cg = cooperative_groups;
using vec3 = glm::fvec3;
using vec4 = glm::fvec4;
using mat3 = glm::fmat3;


struct Indices {
    uint32_t block_size, tile_id;
    uint32_t i, j, pix_id;
    bool inside;
};


struct IndexingData {
    uint32_t N, n_isects;
    uint32_t image_width, image_height;
    uint32_t tile_width, tile_height;
    uint32_t num_tiles_width, num_tiles_height;

    [[nodiscard]] DEVICE Indices get_indices(const cg::thread_block& block) const {
        uint32_t i = block.group_index().y * tile_height + block.thread_index().y;
        uint32_t j = block.group_index().z * tile_width + block.thread_index().x;

        return {
            .block_size = block.size(),
            .tile_id = block.group_index().y * num_tiles_width + block.group_index().z,
            .i = i,
            .j = j,
            .pix_id = min(i * image_width + j, image_width * image_height - 1),
            .inside = (i < image_height && j < image_width),
        };
    }
};


struct IntersectionRange {
    int32_t range_start, range_end;
    uint32_t num_batches;

    DEVICE IntersectionRange(const IndexingData& data, const Indices& indices, const int32_t* tile_offsets) {
        range_start = tile_offsets[indices.tile_id];
        bool is_last_tile = (indices.tile_id == data.num_tiles_width * data.num_tiles_height - 1);
        range_end = is_last_tile ? static_cast<int32_t>(data.n_isects) : tile_offsets[indices.tile_id + 1];
        num_batches = (range_end - range_start + indices.block_size - 1) / indices.block_size;
    };
};


inline __device__ mat3 quat_to_rotmat(const vec4 quat) {
    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    // normalize
    float inv_norm = rsqrt(x * x + y * y + z * z + w * w);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
    w *= inv_norm;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;
    return mat3(
        (1.f - 2.f * (y2 + z2)),
        (2.f * (xy + wz)),
        (2.f * (xz - wy)), // 1st col
        (2.f * (xy - wz)),
        (1.f - 2.f * (x2 + z2)),
        (2.f * (yz + wx)), // 2nd col
        (2.f * (xz + wy)),
        (2.f * (yz - wx)),
        (1.f - 2.f * (x2 + y2)) // 3rd col
    );
}

inline __device__ vec3 safe_normalize(vec3 v) {
    const float l = v.x * v.x + v.y * v.y + v.z * v.z;
    return l > 0.0f ? (v * rsqrtf(l)) : v;
}



template <uint32_t DIM, class WarpT>
DEVICE void warpSum(float *val, WarpT &warp) {
#pragma unroll
    for (uint32_t i = 0; i < DIM; i++) {
        val[i] = cg::reduce(warp, val[i], cg::plus<float>());
    }
}


template <class WarpT> inline __device__ void warpSum(float &val, WarpT &warp) {
    val = cg::reduce(warp, val, cg::plus<float>());
}

template <class WarpT> inline __device__ void warpSum(glm::fvec4 &val, WarpT &warp) {
    val.x = cg::reduce(warp, val.x, cg::plus<float>());
    val.y = cg::reduce(warp, val.y, cg::plus<float>());
    val.z = cg::reduce(warp, val.z, cg::plus<float>());
    val.w = cg::reduce(warp, val.w, cg::plus<float>());
}

template <class WarpT> inline __device__ void warpSum(glm::fvec3 &val, WarpT &warp) {
    val.x = cg::reduce(warp, val.x, cg::plus<float>());
    val.y = cg::reduce(warp, val.y, cg::plus<float>());
    val.z = cg::reduce(warp, val.z, cg::plus<float>());
}

template <class WarpT> inline __device__ void warpSum(glm::fvec2 &val, WarpT &warp) {
    val.x = cg::reduce(warp, val.x, cg::plus<float>());
    val.y = cg::reduce(warp, val.y, cg::plus<float>());
}

template <class WarpT> inline __device__ void warpSum(glm::fmat4 &val, WarpT &warp) {
    warpSum(val[0], warp);
    warpSum(val[1], warp);
    warpSum(val[2], warp);
    warpSum(val[3], warp);
}

template <class WarpT> inline __device__ void warpSum(glm::fmat3 &val, WarpT &warp) {
    warpSum(val[0], warp);
    warpSum(val[1], warp);
    warpSum(val[2], warp);
}

template <class WarpT> inline __device__ void warpSum(glm::fmat2 &val, WarpT &warp) {
    warpSum(val[0], warp);
    warpSum(val[1], warp);
}


DEVICE void atomicAdd(float* dst, const glm::fvec3& src) {
    gpuAtomicAdd(dst, src.x);
    gpuAtomicAdd(dst + 1, src.y);
    gpuAtomicAdd(dst + 2, src.z);
}
DEVICE void atomicAdd(float* dst, const glm::fvec4& src) {
    gpuAtomicAdd(dst, src.x);
    gpuAtomicAdd(dst + 1, src.y);
    gpuAtomicAdd(dst + 2, src.z);
    gpuAtomicAdd(dst + 3, src.w);
}

template <uint32_t CDIM>
DEVICE void atomicAdd(float* dst, const float* src) {
#pragma unroll
    for (uint32_t k = 0; k < CDIM; ++k) {
        gpuAtomicAdd(dst + k, src[k]);
    }
}


template <uint32_t CDIM>
HOSTDEVICE void blend_from(float* dst_ptr, const float* src_ptr, const float factor) {
    if (src_ptr == nullptr) {return;}
#pragma unroll
    for (uint32_t k = 0; k < CDIM; ++k) {
        dst_ptr[k] += src_ptr[k] * factor;
    }
}
template <uint32_t CDIM>
HOSTDEVICE void blend_from(float* dst_ptr, const float* src_ptr) {
    if (src_ptr == nullptr) {return;}
#pragma unroll
    for (uint32_t k = 0; k < CDIM; ++k) {
        dst_ptr[k] += src_ptr[k];
    }
}

template <uint32_t CDIM>
HOSTDEVICE void set_from(float* dst_ptr, const float* src_ptr) {
    if (src_ptr == nullptr) {return;}
#pragma unroll
    for (uint32_t k = 0; k < CDIM; ++k) {
        dst_ptr[k] = src_ptr[k];
    }
}

template <uint32_t CDIM>
HOSTDEVICE void set_from(float* dst_ptr, const float* src_ptr, const float factor) {
    if (src_ptr == nullptr) {return;}
#pragma unroll
    for (uint32_t k = 0; k < CDIM; ++k) {
        dst_ptr[k] = src_ptr[k] * factor;
    }
}

DEVICE vec3 safe_normalize_bwd(const vec3 &v, const vec3 &d_out) {
    const float l = v.x * v.x + v.y * v.y + v.z * v.z;
    if (l > 0.0f) {
        const float il = rsqrtf(l);
        const float il3 = (il * il * il);
        return il * d_out - il3 * glm::dot(d_out, v) * v;
    }
    return d_out;
}

DEVICE void quat_to_rotmat_vjp(const vec4 quat, const mat3 v_R, vec4 &v_quat) {
    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    // normalize
    float inv_norm = rsqrt(x * x + y * y + z * z + w * w);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
    w *= inv_norm;
    vec4 v_quat_n = vec4(
        2.f * (x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
               z * (v_R[0][1] - v_R[1][0])),
        2.f *
            (-2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
             z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])),
        2.f * (x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
               z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])),
        2.f * (x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
               2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0]))
    );

    vec4 quat_n = vec4(w, x, y, z);
    v_quat += (v_quat_n - glm::dot(v_quat_n, quat_n) * quat_n) * inv_norm;
}

DEVICE void quat_scale_to_preci_half_vjp(
    // fwd inputs
    const vec4 quat,
    const vec3 scale,
    // precompute
    const mat3 R,
    // grad outputs
    const mat3 v_M, // M is perci_half
    // grad inputs
    vec4 &v_quat,
    vec3 &v_scale
) {
    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    float sx = 1.0f / scale[0], sy = 1.0f / scale[1], sz = 1.0f / scale[2];

    // M = R * S
    mat3 S = mat3(sx, 0.f, 0.f, 0.f, sy, 0.f, 0.f, 0.f, sz);
    mat3 v_R = v_M * S;

    // grad for (quat, scale) from preci
    quat_to_rotmat_vjp(quat, v_R, v_quat);

    v_scale[0] +=
        -sx * sx *
        (R[0][0] * v_M[0][0] + R[0][1] * v_M[0][1] + R[0][2] * v_M[0][2]);
    v_scale[1] +=
        -sy * sy *
        (R[1][0] * v_M[1][0] + R[1][1] * v_M[1][1] + R[1][2] * v_M[1][2]);
    v_scale[2] +=
        -sz * sz *
        (R[2][0] * v_M[2][0] + R[2][1] * v_M[2][1] + R[2][2] * v_M[2][2]);
}



#endif //XSIMGS_RASTERIZAITON_UTILS_H
