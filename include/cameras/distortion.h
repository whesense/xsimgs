#ifndef XSIMGS_DISTORTION_H
#define XSIMGS_DISTORTION_H

#include <tuple>
#include <toast/utils/vector_math.h>


HOSTDEVICE std::tuple<float, float2> uv_distortion(
     const float2& uv,
     const float4& dist0,
     const float4& dist1
) {
    float2 uv_sq{uv.x * uv.x, uv.y * uv.y};
    float r2 = uv_sq.x + uv_sq.y;

    const auto& [k1, k2, p1, p2] = dist0;
    const auto& [k3, k4, k5, k6] = dist1;

    const float numerator = 1.0f + r2 * (k1 + r2 * (k2 + r2 * k3));
    const float denom = 1.0f + r2 * (k4 + r2 * (k5 + r2 * k6));
    const float spherical_scale = numerator / denom;

    const float uv2 = 2.0f * uv.x * uv.y;
    const float2 uv_delta{
        p1 * uv2                   + p2 * (r2 + 2.0f * uv_sq.x),
        p1 * (r2 + 2.0f * uv_sq.y) + p2 * uv2
   };

    return {spherical_scale, uv_delta};
}


template <typename CameraType>
torch::Tensor undistort_map(
    const torch::Tensor& camera_pack,
    uint width,
    uint height,
    bool normalized_range
) {
    validate_camera_pack<CameraType>(camera_pack);
    auto out = torch::zeros({height, width, 2}, camera_pack.options());

    auto data = std::make_tuple(
        (float2*)out.data_ptr(), (CameraType*)camera_pack.data_ptr(),
        uint2{width, height}, normalized_range);

    INLINE_KERNEL(width * height, camera_pack.device(), {
        const auto [out_ptr, cam_ptr, size, norm_range] = data;
        const auto px = cam_ptr->pixel_idx(idx, size);
        const auto uv = cam_ptr->pixel_to_uv(px, size);
        auto result = cam_ptr->impl.distort_uv(uv);
        out_ptr[idx] = norm_range ? result : result * 2.0f - 1.0f;
    })

    return out;
}



#endif //XSIMGS_DISTORTION_H
