#include <toast/utils/vector_math.h>
#include <cameras/bind.h>


template <uint max_undistort_iterations = 20>
struct OpenCVFisheyeCameraImpl: BaseCameraImpl {
    constexpr static std::array<int, 2> tensor_dims = {4, 4};

    float2 f;
    float2 c;
    float4 dist; // k1, k2, k3, k4


    HOSTDEVICE float fisheye_distortion(const float theta) const {
        const auto [k1, k2, k3, k4] = dist;
        const auto theta2 = theta * theta;
        auto delta = theta2 * k4 + k3;
        delta = theta2 * delta + k2;
        delta = theta2 * delta + k1;
        delta = theta2 * delta + 1.0f;
        return theta * delta;
    }

    HOSTDEVICE float fisheye_distortion_deriv(const float theta) const {
        const auto [k1, k2, k3, k4] = dist;
        const auto theta2 = theta * theta;
        auto d = theta2 * 9.0f * k4 + 7.0f * k3;
        d = theta2 * d + 5.0f * k2;
        d = theta2 * d + 3.0f * k1;
        d = theta2 * d + 1.0f;
        return d;
    }

    HOSTDEVICE float solve_theta(const float delta, float& residual) const {
        float theta = delta;
        residual = 0.0f;
        #pragma unroll
        for (uint i = 0; i < max_undistort_iterations; ++i) {
            residual = fisheye_distortion(theta) - delta;
            // derivative clamp keeps the iteration stable
            // near and past the monotonic limit of the distortion polynomial
            theta -= residual / fmaxf(fisheye_distortion_deriv(theta), 0.1f);
            theta = fminf(fmaxf(theta, 0.0f), float(M_PI));
            if (fabsf(residual) < 1e-6f) break;
        }
        return theta;
    }

    HOSTDEVICE float3 pixel_to_camera_ray(const float2& uv) const {
        float2 uv_cam = (uv - c) / f;
        const auto delta = length(uv_cam);

        float residual;
        const float theta = solve_theta(delta, residual);

        if (theta <= 0.0f) {
            return {0.0f, 0.0f, 1.0f};
        }

        float theta_sin, theta_cos;
        sincosf(theta, &theta_sin, &theta_cos);
        const float scale_factor = theta_sin / delta;
        return {scale_factor * uv_cam.x, scale_factor * uv_cam.y, theta_cos};
    }

    HOSTDEVICE bool ray_valid(const float2& uv) const {
        const float delta = length((uv - c) / f);
        if (delta <= 1e-8f) return true;  // image center: exact on-axis ray
        float residual;
        const float theta = solve_theta(delta, residual);
        // converged, and on the monotonic branch of the polynomial
        return (fabsf(residual) < 1e-5f) &&
               (fisheye_distortion_deriv(theta) > 0.0f);
    }

    template <typename SHMemType>
    __device__ __forceinline__ void pixel_to_camera_ray_backward(
        const float2& uv,
        const float3& d_ray,
        char* block_outputs,
        SHMemType& sh_mem,
        uint base_offset
    ) const {
        const float2 uv_cam = (uv - c) / f;
        const float delta = length(uv_cam);

        // re-solve theta_d(theta) = delta, as in the forward pass
        float residual;
        const float theta = solve_theta(delta, residual);
        const float deriv = fisheye_distortion_deriv(theta);

        float4 d_fc{0.0f, 0.0f, 0.0f, 0.0f};
        float4 d_dist{0.0f, 0.0f, 0.0f, 0.0f};

        const bool valid =
            (theta > 0.0f) && (delta > 1e-8f) &&
            (fabsf(residual) < 1e-5f) && (deriv > 1e-3f);

        if (valid) {
            float theta_sin, theta_cos;
            sincosf(theta, &theta_sin, &theta_cos);
            const float inv_delta = 1.0f / delta;
            const float s = theta_sin * inv_delta;

            // ray = (s * uv_cam, cos(theta)), s = sin(theta) / delta
            const float d_s = d_ray.x * uv_cam.x + d_ray.y * uv_cam.y;
            const float d_theta =
                d_s * theta_cos * inv_delta - d_ray.z * theta_sin;

            // theta(delta, k) is implicit through theta_d(theta, k) = delta:
            // dtheta/ddelta = 1/theta_d', dtheta/dk_i = -theta^(2i+1)/theta_d'
            const float d_theta_scaled = d_theta / deriv;
            const float theta2 = theta * theta;
            const float theta3 = theta2 * theta;
            d_dist = {
                -d_theta_scaled * theta3,
                -d_theta_scaled * theta3 * theta2,
                -d_theta_scaled * theta3 * theta2 * theta2,
                -d_theta_scaled * theta3 * theta2 * theta2 * theta2
            };

            const float d_delta =
                -d_s * theta_sin * inv_delta * inv_delta + d_theta_scaled;
            const float2 d_uv_cam{
                s * d_ray.x + d_delta * uv_cam.x * inv_delta,
                s * d_ray.y + d_delta * uv_cam.y * inv_delta
            };

            // uv_cam = (uv - c) / f
            d_fc = {
                -d_uv_cam.x * uv_cam.x / f.x,
                -d_uv_cam.y * uv_cam.y / f.y,
                -d_uv_cam.x / f.x,
                -d_uv_cam.y / f.y
            };
        }

        block_reduce_float4(block_outputs, base_offset, d_fc)
        block_reduce_float4(block_outputs,
            base_offset + offsetof(OpenCVFisheyeCameraImpl, dist), d_dist)
    }

    HOSTDEVICE float3 camera_point_to_uvd(float3 point) const {
        const auto norm = fmaxf(1e-6f, hypotf(point.x, point.y));
        const auto theta = atan2f(norm, point.z);

        const bool valid = fisheye_distortion_deriv(theta) > 0.0f;
        const auto delta = fisheye_distortion(theta) / norm;

        const auto uv = f * delta * float2{point.x, point.y} + c;
        const auto depth = (valid && (delta > 0.0f)) ? length(point) : 0.0f;
        return {uv.x, uv.y, depth};
    }

    HOSTDEVICE float2 distort_uv(const float2& uv) const {
        const float2 uv_cam = (uv - c) / f;
        const float theta = length(uv_cam);
        // theta_d(theta) / theta -> 1 as theta -> 0
        const float scale = (theta > 1e-8f) ? fisheye_distortion(theta) / theta : 1.0f;
        return uv_cam * scale * f + c;
    }

    HOSTDEVICE void lerp(
        OpenCVFisheyeCameraImpl& result,
        const OpenCVFisheyeCameraImpl& other,
        float t
    ) const {
        result.f = ::lerp(f, other.f, t);
        result.c = ::lerp(c, other.c, t);
        result.dist = ::lerp(dist, other.dist, t);
    }

    HOSTDEVICE static OpenCVFisheyeCameraImpl pack(const char** data, const uint idx) {
        const auto params = reinterpret_cast<const float4*>(data[0])[idx];
        const float4* dist_ptr = reinterpret_cast<const float4*>(data[1]);
        return {
            .f = {params.x, params.y},
            .c = {params.z, params.w},
            .dist = dist_ptr[idx],
        };
    }

    HOSTDEVICE void unpack(char** data, uint idx) const {
        reinterpret_cast<float4*>(data[0])[idx] = {f.x, f.y, c.x, c.y};
        auto dist_ptr = reinterpret_cast<float4*>(data[1]);
        dist_ptr[idx] = dist;
    }
};


void bind_fisheye(pybind11::module& m) {
    using camera_t = Camera<OpenCVFisheyeCameraImpl<>>;
    pybind_camera<camera_t>(m, "OpenCVFisheyeCamera");
}
