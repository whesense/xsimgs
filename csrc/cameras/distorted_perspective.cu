#include <toast/utils/vector_math.h>
#include <cameras/bind.h>
#include <cameras/distortion.h>


template <uint num_undistort_iterations = 5>
struct DistortedPerspectiveCameraImpl: BaseCameraImpl {
    constexpr static std::array<int, 3> tensor_dims = {4, 8, 1};
    constexpr static float min_jacobian_det = 1e-3f;

    struct DistortionState {
        float scale;
        float denom;  // denominator of the rational radial scale
        float2 delta;
        float j_uu;
        float j_uv;
        float j_vv;

        [[nodiscard]] HOSTDEVICE float det() const {
            return j_uu * j_vv - j_uv * j_uv;
        }

        [[nodiscard]] HOSTDEVICE float2 apply(const float2& uv) const {
            return uv * scale + delta;
        }

        [[nodiscard]] HOSTDEVICE float2 pullback(const float2& d_uv) const {
            const float d = det();
            if (fabsf(d) <= min_jacobian_det) {
                return d_uv / scale;
            }

            return {
                (j_vv * d_uv.x - j_uv * d_uv.y) / d,
                (j_uu * d_uv.y - j_uv * d_uv.x) / d
            };
        }
    };

    float2 f;
    float2 c;
    float4 dist0;
    float4 dist1;

    float max_radius_sq;


    HOSTDEVICE DistortionState distortion_state(const float2& uv) const {
        const auto& [k1, k2, p1, p2] = dist0;
        const auto& [k3, k4, k5, k6] = dist1;

        const float u = uv.x;
        const float v = uv.y;
        const float r2 = u * u + v * v;

        const float num = 1.0f + r2 * (k1 + r2 * (k2 + r2 * k3));
        const float denom = 1.0f + r2 * (k4 + r2 * (k5 + r2 * k6));
        const float scale = num / denom;

        // d(scale)/d(r2), by the quotient rule
        const float d_num = k1 + r2 * (2.0f * k2 + 3.0f * k3 * r2);
        const float d_den = k4 + r2 * (2.0f * k5 + 3.0f * k6 * r2);
        const float d_scale = (d_num * denom - num * d_den) / (denom * denom);

        return {
            .scale = scale,
            .denom = denom,
            .delta = {
                2.0f * p1 * u * v + p2 * (r2 + 2.0f * u * u),
                p1 * (r2 + 2.0f * v * v) + 2.0f * p2 * u * v
            },
            .j_uu = scale + 2.0f * u * u * d_scale + 2.0f * p1 * v + 6.0f * p2 * u,
            .j_uv = 2.0f * u * v * d_scale + 2.0f * p1 * u + 2.0f * p2 * v,
            .j_vv = scale + 2.0f * v * v * d_scale + 6.0f * p1 * v + 2.0f * p2 * u
        };
    }

    HOSTDEVICE float2 undistort_uv(const float2& uv_distorted, DistortionState& state) const {
        float2 uv = uv_distorted;

        #pragma unroll
        for (uint j = 0; j < num_undistort_iterations; ++j) {
            state = distortion_state(uv);

            const float2 residual = state.apply(uv) - uv_distorted;
            const float det = state.det();

            if (fabsf(det) > min_jacobian_det) {
                uv.x -= (state.j_vv * residual.x - state.j_uv * residual.y) / det;
                uv.y -= (state.j_uu * residual.y - state.j_uv * residual.x) / det;
            } else {
                uv = (uv_distorted - state.delta) / state.scale;
            }
        }

        return uv;
    }

    HOSTDEVICE float3 pixel_to_camera_ray(const float2& uv) const {
        DistortionState state;
        const float2 uv_cam = undistort_uv((uv - c) / f, state);

        return {uv_cam.x, uv_cam.y, 1.0f};
    }

    HOSTDEVICE bool ray_valid(const float2& uv) const {
        DistortionState state;
        const float2 uv_distorted = (uv - c) / f;
        const float2 uv_cam = undistort_uv(uv_distorted, state);

        const float2 residual = state.apply(uv_cam) - uv_distorted;
        const float r2 = uv_cam.x * uv_cam.x + uv_cam.y * uv_cam.y;
        return (r2 <= max_radius_sq) &&
               (fabsf(residual.x) < 1e-5f) && (fabsf(residual.y) < 1e-5f);
    }

    // Gradients w.r.t. the distortion coefficients (dist0, dist1) at a solved
    // uv_cam, by the implicit function theorem: D(uv_cam; k) = uv_distorted gives
    // d(uv_cam)/dk = -J^{-1} dD/dk, so with g_d = J^{-T} g (state.pullback)
    //   dL/dk = -g_d . dD/dk, where (u, v) = uv_cam, r2 = u^2 + v^2:
    //   dD/dk_{1,2,3} =  uv_cam * r2^{1,2,3} / den
    //   dD/dk_{4,5,6} = -uv_cam * scale * r2^{1,2,3} / den
    //   dD/dp1 = (2 u v, r2 + 2 v^2),  dD/dp2 = (r2 + 2 u^2, 2 u v)
    HOSTDEVICE std::tuple<float4, float4> distortion_coefficients_backward(
        const float2& uv_cam,
        const DistortionState& state,
        const float2& d_uv_distorted
    ) const {
        const float u = uv_cam.x;
        const float v = uv_cam.y;
        const float r2 = u * u + v * v;
        const float r4 = r2 * r2;
        const float r6 = r4 * r2;

        // the radial scale moves uv_cam along itself, so only g_d . uv_cam matters
        const float d_radial = -(d_uv_distorted.x * u + d_uv_distorted.y * v) / state.denom;
        const float d_radial_den = -d_radial * state.scale;

        const float4 d_dist0 = {
            d_radial * r2,
            d_radial * r4,
            -(d_uv_distorted.x * 2.0f * u * v + d_uv_distorted.y * (r2 + 2.0f * v * v)),
            -(d_uv_distorted.x * (r2 + 2.0f * u * u) + d_uv_distorted.y * 2.0f * u * v)
        };
        const float4 d_dist1 = {
            d_radial * r6,
            d_radial_den * r2,
            d_radial_den * r4,
            d_radial_den * r6
        };
        return {d_dist0, d_dist1};
    }

    template <typename SHMemType>
    __device__ __forceinline__ void pixel_to_camera_ray_backward(
        const float2& uv,
        const float3& d_ray,
        char* block_outputs,
        SHMemType& sh_mem,
        uint base_offset
    ) const {
        const float2 inv_f{1.0f / f};
        const float2 uv_distorted = (uv - c) * inv_f;

        // re-solve D(uv_cam) = uv_distorted as in the forward pass, then take the
        // distortion state at the solution itself so that the Jacobian and the
        // coefficient derivatives below are evaluated at the same point
        DistortionState state;
        const float2 uv_cam = undistort_uv(uv_distorted, state);
        state = distortion_state(uv_cam);

        // gradient w.r.t. uv_distorted: J^{-T} d_ray.xy (J is symmetric); the
        // degenerate-Jacobian fallback of pullback (J ~ scale * I) is shared with
        // the coefficient gradients through the same d_uv_distorted
        const float2 d_uv_distorted = state.pullback({d_ray.x, d_ray.y});

        // uv_distorted = (uv - c) / f
        const float4 d_fc = {
            -d_uv_distorted.x * uv_distorted.x * inv_f.x,
            -d_uv_distorted.y * uv_distorted.y * inv_f.y,
            -d_uv_distorted.x * inv_f.x,
            -d_uv_distorted.y * inv_f.y
        };
        const auto [d_dist0, d_dist1] =
            distortion_coefficients_backward(uv_cam, state, d_uv_distorted);

        block_reduce_float4(block_outputs, base_offset, d_fc)
        block_reduce_float4(block_outputs,
            base_offset + offsetof(DistortedPerspectiveCameraImpl, dist0), d_dist0)
        block_reduce_float4(block_outputs,
            base_offset + offsetof(DistortedPerspectiveCameraImpl, dist1), d_dist1)
    }

    HOSTDEVICE float3 camera_point_to_uvd(float3 point) const {
        const float2 xy_{point.x / point.z, point.y / point.z};
        const auto [scale, delta] = uv_distortion(xy_, dist0, dist1);
        const auto [u, v] = (xy_ * scale + delta) * f + c;

        const float r2 = xy_.x * xy_.x + xy_.y * xy_.y;
        return {u, v, (r2 <= max_radius_sq) ? point.z : 0.0f};
    }

    HOSTDEVICE float2 distort_uv(const float2& uv) const {
        const auto uv_d = (uv - c) / f;
        const auto [scale, delta] = uv_distortion(uv_d, dist0, dist1);
        return (uv_d * scale + delta) * f + c;
    }

    HOSTDEVICE void lerp(
        DistortedPerspectiveCameraImpl& result,
        const DistortedPerspectiveCameraImpl& other,
        float t
    ) const {
        result.f = ::lerp(f, other.f, t);
        result.c = ::lerp(c, other.c, t);
        result.dist0 = ::lerp(dist0, other.dist0, t);
        result.dist1 = ::lerp(dist1, other.dist1, t);
        result.max_radius_sq = ::lerp(max_radius_sq, other.max_radius_sq, t);
    }

    HOSTDEVICE static DistortedPerspectiveCameraImpl pack(const char** data, const uint idx) {
        const auto params = reinterpret_cast<const float4*>(data[0])[idx];
        const float4* dist_ptr = reinterpret_cast<const float4*>(data[1]);
        return {
            .f = {params.x, params.y},
            .c = {params.z, params.w},
            .dist0 = dist_ptr[2*idx],
            .dist1 = dist_ptr[2*idx+1],
            .max_radius_sq = reinterpret_cast<const float*>(data[2])[idx],
        };
    }

    HOSTDEVICE void unpack(char** data, uint idx) const {
        reinterpret_cast<float4*>(data[0])[idx] = {f.x, f.y, c.x, c.y};
        auto dist_ptr = reinterpret_cast<float4*>(data[1]);
        dist_ptr[2*idx] = dist0;
        dist_ptr[2*idx+1] = dist1;
        reinterpret_cast<float*>(data[2])[idx] = max_radius_sq;
    }
};


void bind_distorted_perspective(pybind11::module& m) {
    pybind_camera<Camera<DistortedPerspectiveCameraImpl<>>>(m, "DistortedPerspectiveCamera");
}
