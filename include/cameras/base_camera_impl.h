#ifndef XSIMGS_BASE_CAMERA_IMPL_H
#define XSIMGS_BASE_CAMERA_IMPL_H

#include <array>

#include "camera.h"

struct BaseCameraImpl {
    [[nodiscard]] HOSTDEVICE float2 pixel_to_uv(int2 px, uint2 size) const {
        return {
            (static_cast<float>(px.x) + 0.5f) / static_cast<float>(size.x),
            (static_cast<float>(px.y) + 0.5f) / static_cast<float>(size.y)
        };
    }

    // Implement one of:
    // HOSTDEVICE float3 pixel_to_camera_ray(int2 px, uint2 size) const;
    // HOSTDEVICE float3 pixel_to_camera_ray(const float2& uv) const;

    // Implement one of:
    template <typename T> DEVICE void pixel_to_camera_ray_backward(
       const float2& uv, const float3& d_ray, char* block_outputs,
       T& sh_mem, uint base_offset) const {}
    // template <typename T> DEVICE void pixel_to_camera_ray_backward(
    //    const int2& px, const int2& size, const float3& d_ray,
    //    char* block_outputs, T& sh_mem, uint base_offset) const;

    // Implement one of:
    // HOSTDEVICE float3 camera_point_to_uvd(float3 point) const;
    // HOSTDEVICE void camera_point_to_uvd_jacob(
    //     const float3& point, const float3& d_point,
    //     float3& uvd, float2& du_dv
    // ) const;

    // Add camera parameters (intrinsics) to these variables/methods
    constexpr static std::array<int, 0> tensor_dims = {};

    HOSTDEVICE void lerp(
        BaseCameraImpl& result, const BaseCameraImpl& other, float t) const {}

    HOSTDEVICE static BaseCameraImpl pack(const char** data, const uint idx) {
        return {};
    }
    HOSTDEVICE void unpack(char** data, uint idx) const {}
};

#endif //XSIMGS_BASE_CAMERA_IMPL_H
