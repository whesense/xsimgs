#ifndef XSIMGS_SPHERICAL_H
#define XSIMGS_SPHERICAL_H


#include <cameras/camera.h>
#include <cameras/base_camera_impl.h>


HOSTDEVICE float3 spherical_to_cartesian(const float3& polar) {
    const auto& [azimuth, elevation, distance] = polar;

    float elevation_sin, elevation_cos, azimuth_sin, azimuth_cos;
    sincosf(elevation, &elevation_sin, &elevation_cos);
    sincosf(azimuth, &azimuth_sin, &azimuth_cos);
    const auto xy_dist = distance * elevation_cos;

    return {
        xy_dist * azimuth_sin,
        xy_dist * azimuth_cos,
        distance * elevation_sin
    };
}

HOSTDEVICE float3 cartesian_to_spherical(const float3& p) {
    const float distance = length(p);
    return {
        atan2f(-p.x, -p.y) + float(M_PI),
        asinf(p.z / distance),
        distance
    };
}


struct SphericalCameraImpl: BaseCameraImpl {
    constexpr static std::array<int, 1> tensor_dims = {4,};

    float phi_min, phi_range;
    float theta_min, theta_range;

    HOSTDEVICE float3 pixel_to_camera_ray(const float2& uv) const {
        return spherical_to_cartesian({
            phi_min + uv.x * phi_range,
            theta_min + uv.y * theta_range,
            1.0f
        });
    }

    HOSTDEVICE void lerp(
        SphericalCameraImpl& result,
        const SphericalCameraImpl& other,
        float t
    ) const {
        result.phi_min = ::lerp(phi_min, other.phi_min, t);
        result.phi_range = ::lerp(phi_range, other.phi_range, t);
        result.theta_min = ::lerp(theta_min, other.theta_min, t);
        result.theta_range = ::lerp(theta_range, other.theta_range, t);
    }

    HOSTDEVICE static SphericalCameraImpl pack(const char** data, const uint idx) {
        const auto params = reinterpret_cast<const float4*>(data[0])[idx];
        return {
            .phi_min = params.x, .phi_range = params.y,
            .theta_min = params.z, .theta_range = params.w
        };
    }

    HOSTDEVICE void unpack(char** data, uint idx) const {
        reinterpret_cast<float4*>(data[0])[idx] = {phi_min, phi_range, theta_min, theta_range};
    }
};


struct LidarCameraImpl: public SphericalCameraImpl {
    constexpr static std::array<int, 2> tensor_dims = {4, 2};
    const float* elev_values;

    HOSTDEVICE float2 pixel_to_uv(const int2& px, const uint2& size) const {
        return {
            (static_cast<float>(px.x) + 0.5f) / static_cast<float>(size.x),
            elev_values[px.y]
        };
    }

    HOSTDEVICE static LidarCameraImpl pack(const char** data, const uint idx) {
        const auto params = reinterpret_cast<const float4*>(data[0])[idx];
        const auto elev_ptr = static_cast<const float**>((void*)data[1])[idx];
        return {
            SphericalCameraImpl{
                .phi_min = params.x, .phi_range = params.y,
                .theta_min = params.z, .theta_range = params.w
            },
            elev_ptr
        };
    }

    HOSTDEVICE void unpack(char** data, uint idx) const {
        reinterpret_cast<float4*>(data[0])[idx] = {phi_min, phi_range, theta_min, theta_range};
        static_cast<const float**>((void*)data[1])[idx] = elev_values;
    }
};


template <typename T = SphericalCameraImpl>
struct SphericalCamera : public Camera<T> {
    constexpr static bool with_phase_modeling = true;

    HOSTDEVICE explicit SphericalCamera(const Camera<T>& obj): Camera<T>(obj) {}

    HOSTDEVICE float2 angles_to_uv(ProjectionMode mode, const float phi, const float theta) const {
        float u = (phi - Camera<T>::impl.phi_min) / Camera<T>::impl.phi_range;

        const float u_pose = (fabsf(Camera<T>::rolling_time.x) > 0.0f) ?
            (Camera<T>::pose_time - Camera<T>::trigger_time) / Camera<T>::rolling_time.x : 0.5f;
        const float u_offset = (2.0f * float(M_PI) / fabsf(Camera<T>::impl.phi_range)); // todo: check for negative phi_range
        if (Camera<T>::rolling_time.x > 0) {
            if ((mode == ProjectionMode::NegativeDelta) && (u > u_pose)) u = u - u_offset;
            if ((mode == ProjectionMode::PositiveDelta) && (u < u_pose)) u = u + u_offset;
        } else {
            if ((mode == ProjectionMode::NegativeDelta) && (u < u_pose)) u = u + u_offset;
            if ((mode == ProjectionMode::PositiveDelta) && (u > u_pose)) u = u - u_offset;
        }

        const float v = (theta - Camera<T>::impl.theta_min) / Camera<T>::impl.theta_range;
        return {u, v};
    }

    HOSTDEVICE float3 camera_point_to_uvd(ProjectionMode mode, float3 point) const {
        const auto [phi, theta, dist] = cartesian_to_spherical(point);
        const auto [u, v] = angles_to_uv(mode, phi, theta);
        return {u, v, dist};
    }

    HOSTDEVICE void camera_point_to_uvd_jacob(
        ProjectionMode mode, const float3& point, const float3& d_point,
        float3& uvd, float2& du_dv
    ) const {
        const float x2y2 = point.x * point.x + point.y * point.y;
        const float r = sqrtf(x2y2 + point.z * point.z);

        const float phi = atan2f(-point.x, -point.y) + M_PI;
        const float theta = asinf(point.z / r);

        const auto [u, v] = angles_to_uv(mode, phi, theta);
        uvd = {u, v, r};

        const float x2y2_sqrt = sqrtf(x2y2);
        const float r2 = r * r;
        const float mul = point.z / (r2 * x2y2_sqrt);
        du_dv = {
            ((-point.x * d_point.y + point.y * d_point.x) / x2y2) / Camera<T>::impl.phi_range,
            ((x2y2_sqrt * d_point.z) / r2 - mul * (point.x * d_point.x + point.y * d_point.y)) / Camera<T>::impl.theta_range
        };
    }


    HOSTDEVICE static SphericalCamera pack(const char** data, const uint idx) {
        return SphericalCamera(Camera<T>::pack(data, idx));
    }

};

using RegularSphericalCamera = SphericalCamera<>;
using LidarCamera = SphericalCamera<LidarCameraImpl>;

#endif //XSIMGS_SPHERICAL_H
