#ifndef XSIMGS_CAMERAS_H
#define XSIMGS_CAMERAS_H

#include <tuple>

#include <ATen/core/Tensor.h>

#include <toast/utils/vector_math.h>
#include <toast/math/motion.h>
#include <toast/utils/launch_helpers.h>
#include <toast/utils/tensor_helpers.h>

#include <utils/reduce_helpers.cuh>

enum class ProjectionMode: int {
    Central = 0,
    NegativeDelta = 1,
    PositiveDelta = 2
};

#define _arg(t) std::declval<t>()
#define define_has_member_named(name, member, ...) \
    template <typename, typename = std::void_t<>>\
    struct has_##name : std::false_type {};\
    template <typename T>\
    struct has_##name<T, std::void_t<decltype(std::declval<T&>().member(__VA_ARGS__))>> : std::true_type {};\
    template <typename T>\
    inline constexpr bool has_##name##_v = has_##name<T>::value;

#define define_has_member(member, ...) \
    define_has_member_named(member, member, __VA_ARGS__)


define_has_member_named(
    camera_point_to_uvd_jacob_impl,
    camera_point_to_uvd_jacob,
    _arg(float3),
    _arg(float3),
    _arg(float3&),
    _arg(float2&))

define_has_member(
    camera_point_to_uvd_jacob,
    _arg(ProjectionMode),
    _arg(float3),
    _arg(float3),
    _arg(float3&),
    _arg(float2&))
define_has_member(distort_uv, _arg(float2))
define_has_member(ray_valid, _arg(float2))





template <typename CameraType>
void validate_camera_pack(const at::Tensor& camera_pack) {
    TORCH_CHECK(camera_pack.numel() == CameraType::pack_elements(),
        "Only single camera is currently supported");
    TORCH_CHECK(camera_pack.dtype() == at::kFloat, "Camera type must be float32");
    TORCH_CHECK(camera_pack.is_contiguous(), "Camera pack must be contiguous tensor");
}




template <typename CameraImpl>
struct Camera {
    typedef CameraImpl ImplType;

    quatf q0;
    float3 t0;
    float3 v;
    float3 w;
    float2 rolling_time;
    float trigger_time;
    float pose_time;

    CameraImpl impl;

    [[nodiscard]] HOSTDEVICE int2 pixel_idx(const uint px_linear_idx, const uint2 size) const {
        return int2{int(px_linear_idx % size.x), int(px_linear_idx / size.x)};
    }

    [[nodiscard]] HOSTDEVICE float2 pixel_to_uv(int2 px, uint2 size) const {
        return impl.pixel_to_uv(px, size);
    }

    [[nodiscard]] HOSTDEVICE float pixel_time(const float2& uv) const {
        return trigger_time + dot(uv, rolling_time);
    }

    [[nodiscard]] HOSTDEVICE float pixel_time(const int2& px, const uint2& size) const {
        return pixel_time(pixel_to_uv(px, size));
    }

    [[nodiscard]] HOSTDEVICE float pixel_time_delta(const float2& uv) const {
        return pixel_time(uv) - pose_time;
    }

    [[nodiscard]] HOSTDEVICE std::tuple<quatf, float3> camera_pose_at_time_delta(const float time_delta) const {
        return apply_velocity(q0.std(), t0, v, w, time_delta);
    }

    [[nodiscard]] HOSTDEVICE float3 camera_ray_cs(const float2& uv, bool _normalize = false) const {
        const auto ray = impl.pixel_to_camera_ray(uv);
        return _normalize ? normalize(ray) : ray;
    }

    [[nodiscard]] HOSTDEVICE float3 camera_ray_cs(const int2& px, const uint2& size, bool _normalize = false) const {
        return camera_ray_cs(pixel_to_uv(px, size), _normalize);
    }

    // Whether the ray of this pixel is meaningful. Camera models whose
    // unprojection can fail (e.g. pixels outside a fisheye image circle)
    // implement ray_valid on their impl; everything else is always valid.
    [[nodiscard]] HOSTDEVICE bool ray_valid(const float2& uv) const {
        if constexpr (has_ray_valid_v<CameraImpl>) {
            return impl.ray_valid(uv);
        } else {
            return true;
        }
    }

    [[nodiscard]] HOSTDEVICE std::tuple<float3, float3> camera_ray(const int2& px, const uint2& size, bool _normalize = true) const {
        const float2 uv = pixel_to_uv(px, size);
        const float3 ray = camera_ray_cs(uv, _normalize);
        const float time_delta = pixel_time_delta(uv);
        const auto [q, t] = camera_pose_at_time_delta(time_delta);
        return {t, q.apply(ray)};
    }
 
    [[nodiscard]] HOSTDEVICE auto camera_ray(const uint px_idx, const uint2 size) const {
        return camera_ray(pixel_idx(px_idx, size), size);
    }

    HOSTDEVICE float3 camera_point_to_uvd(ProjectionMode mode, const float3& point) const {
        return impl.camera_point_to_uvd(point);
    }

    constexpr static uint pack_elements() {
        return sizeof(Camera) / sizeof(float);
    }
    constexpr static std::array<int, 7> tensor_dims_base = {4, 3, 3, 3, 2, 1, 1};
    constexpr static bool with_phase_modeling = false;

    [[nodiscard]] HOSTDEVICE Camera lerp(const Camera& other, const float t) const {
        Camera result;
        result.impl = impl.lerp(other.impl, t);
        result.q0 = q0.slerp(other.q0, t);
        result.t0 = lerp(t0, other.t0, t);
        result.v = lerp(v, other.v, t);
        result.w = lerp(w, other.w, t);
        result.rolling_time = lerp(rolling_time, other.rolling_time, t);
        result.trigger_time = lerp(trigger_time, other.trigger_time, t);
        result.pose_time = lerp(pose_time, other.pose_time, t);
        return result;
    }

    HOSTDEVICE static Camera pack(const char** data, const uint idx) {
        return {
            .q0 = reinterpret_cast<const quatf*>(data[0])[idx],
            .t0 = reinterpret_cast<const float3*>(data[1])[idx],
            .v  = reinterpret_cast<const float3*>(data[2])[idx],
            .w  = reinterpret_cast<const float3*>(data[3])[idx],
            .rolling_time = reinterpret_cast<const float2*>(data[4])[idx],
            .trigger_time = reinterpret_cast<const float*>(data[5])[idx],
            .pose_time = reinterpret_cast<const float*>(data[6])[idx],
            .impl = CameraImpl::pack(data + 7, idx)
        };
    }

    HOSTDEVICE void unpack(char **data, const uint idx) const {
        reinterpret_cast<quatf*>(data[0])[idx] = q0;
        reinterpret_cast<float3*>(data[1])[idx] = t0;
        reinterpret_cast<float3*>(data[2])[idx] = v;
        reinterpret_cast<float3*>(data[3])[idx] = w;
        reinterpret_cast<float2*>(data[4])[idx] = rolling_time;
        reinterpret_cast<float*>(data[5])[idx] = trigger_time;
        reinterpret_cast<float*>(data[6])[idx] = pose_time;
        impl.unpack(data + 7, idx);
    }
};


#endif //XSIMGS_CAMERAS_H
