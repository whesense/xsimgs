#include "cameras_bind.h"

#include <toast/utils/vector_math.h>
#include <cameras/bind.h>


struct PerspectiveCameraImpl: BaseCameraImpl {
    constexpr static std::array<int, 1> tensor_dims = {4,};

    float2 f;
    float2 c;

    HOSTDEVICE float3 pixel_to_camera_ray(const float2& uv) const {
        float2 uv_cam = (uv - c) / f;
        return {uv_cam.x, uv_cam.y, 1.0f};
    }

    template <typename SHMemType>
    __device__ __forceinline__ void pixel_to_camera_ray_backward(
        const float2& uv, 
        const float3& d_ray, 
        char* block_outputs,
        SHMemType& sh_mem,
        uint base_offset
    ) const {
        float2 inv_f{1.0f / f};
        float4 d_params = {
            -d_ray.x * (uv.x - c.x) * inv_f.x * inv_f.x,
            -d_ray.y * (uv.y - c.y) * inv_f.y * inv_f.y,
            -d_ray.x * inv_f.x,
            -d_ray.y * inv_f.y
        };
        block_reduce_float4(block_outputs, base_offset, d_params)
    }

    HOSTDEVICE float3 camera_point_to_uvd(float3 point) const {
        return {
            point.x * f.x / point.z + c.x,
            point.y * f.y / point.z + c.y,
            point.z
        };
    }

    HOSTDEVICE void camera_point_to_uvd_jacob(
        const float3& point, const float3& d_point,
        float3& uvd, float2& du_dv
    ) const {
        uvd = camera_point_to_uvd(point);
        const float du_dt = f.x * (d_point.x - uvd.x * d_point.z) / uvd.z;
        const float dv_dt = f.y * (d_point.y - uvd.y * d_point.z) / uvd.z;
        du_dv = {du_dt, dv_dt};
    }

    HOSTDEVICE void lerp(
        PerspectiveCameraImpl& result,
        const PerspectiveCameraImpl& other,
        float t
    ) const {
        result.f = ::lerp(f, other.f, t);
        result.c = ::lerp(c, other.c, t);
    }

    HOSTDEVICE static PerspectiveCameraImpl pack(const char** data, const uint idx) {
        const auto params = reinterpret_cast<const float4*>(data[0])[idx];
        return {.f = {params.x, params.y}, .c = {params.z, params.w}};
    }

    HOSTDEVICE void unpack(char** data, uint idx) const {
        reinterpret_cast<float4*>(data[0])[idx] = {f.x, f.y, c.x, c.y};
    }
};


void bind_perspective(pybind11::module& m) {
    py::enum_<ProjectionMode>(m, "ProjectionMode")
        .value("Central", ProjectionMode::Central)
        .value("NegativeDelta", ProjectionMode::NegativeDelta)
        .value("PositiveDelta", ProjectionMode::PositiveDelta);

    pybind_camera<Camera<PerspectiveCameraImpl>>(m, "PerspectiveCamera");
}
