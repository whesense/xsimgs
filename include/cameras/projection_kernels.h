#ifndef XSIMGS_PROJ_KERNELS_H
#define XSIMGS_PROJ_KERNELS_H

#include <toast/math/motion.h>
#include <toast/math/quat.h>
#include <toast/utils/vector_math.h>


#include "camera.h"


HOSTDEVICE void jac_quat_from_axis_angle(
    const float3& w,
    const float& t,

    float& qw,
    float3& q_xyz,
    float& d_qw,
    float3& d_q_xyz
) {
    const float w_norm2 = dot(w, w);
    const float w_norm = sqrtf(w_norm2);
    const float theta = w_norm * fabsf(t);

    float sin_half, cos_half;
    sincosf(0.5f * theta, &sin_half, &cos_half);

    const float scale = (theta < 1e-2f) ?
            0.5f - theta * theta * (1.0f / 48.0f) : sin_half / theta;

    qw = cos_half;
    q_xyz = scale * w * t;

    d_qw = -0.5f * copysignf(1.0f, t) * sin_half * w_norm;
    d_q_xyz = (0.5f * cos_half) * w;
}

HOSTDEVICE void jac_quat_mul(
    float w1, float3 xyz1,
    float w2, float3 xyz2,
    float dw1, float3 dxyz1,

    float& w, float3& xyz,
    float& dw, float3& dxyz
) {
    w = w1 * w2 - dot(xyz1, xyz2);
    xyz = w1 * xyz2 + w2 * xyz1 + cross(xyz1, xyz2);

    dw = dw1 * w2 - dot(dxyz1, xyz2);
    dxyz = dw1 * xyz2 + w2 * dxyz1 + cross(dxyz1, xyz2);
}

HOSTDEVICE void jac_quat_apply(
    float w, float3 xyz,
    float3 p, float
    dw, float3 dxyz,
    float3& result, float3& dp
) {
    const float3 uv = cross(xyz, p);
    const float3 uuv = cross(xyz, uv);
    result = p + 2.0f * (w * uv + uuv);

    const float3 duv = cross(dxyz, p);
    dp = 2.0f * ( dw * uv + w * duv + cross(dxyz, uv) + cross(xyz, duv) );
}

HOSTDEVICE float3 quat_apply(float w, float3 xyz, float3 p) {
    const float3 uv = cross(xyz, p);
    const float3 uuv = cross(xyz, uv);
    return p + 2.0f * (w * uv + uuv);
}


HOSTDEVICE void jac_world_to_camera(
    const quatf q0,
    const float3 t0,
    const float3 cam_linear_velocity,
    const float3 cam_angular_velocity,

    const float3 point,
    const float3 point_velocity,

    const float time,

    float3& result,
    float3& d_result
) {
    float w1; float3 xyz1; float dw1; float3 dxyz1;
    jac_quat_from_axis_angle(
        cam_angular_velocity, time,
        w1, xyz1,
        dw1, dxyz1);

    float w; float3 xyz; float dw; float3 dxyz;
    jac_quat_mul(
        w1, xyz1,
        q0.real(), q0.img(),
        dw1, dxyz1,
        w, xyz,
        dw, dxyz
        );

    // since we are applying q.inv()
    xyz *= -1;
    dxyz *= -1;

    const float3 velocity_diff = point_velocity - cam_linear_velocity;
    const float3 p = point - t0 + velocity_diff * time;

    float3 dp;
    jac_quat_apply(w, xyz, p, dw, dxyz, result, dp);
    d_result = dp + quat_apply(w, xyz, velocity_diff);
}


template <typename CameraType>
HOSTDEVICE void jac_camera_projection_iteration(
    const CameraType& camera,
    const float3& p_world,
    const float3& p_velocity,
    const ProjectionMode proj_mode,
    float& p_time,
    float3& uvd,
    float& p_time_residual
) {
    // printf("jacobian iteration\n");
    // point_cam = Point coordinates in camera space
    // jac_point_cam = d[point_cam] / d[p_time] - jacobian w.r.t. p_time
    float3 point_cam, jac_point_cam;
    jac_world_to_camera(
        camera.q0, camera.t0, camera.v, camera.w,
        p_world, p_velocity,
        p_time,
        point_cam, jac_point_cam
    );
    // uvd = point projection UV coordinates and depth
    // du_dv = [d[u]/d[p_time], d[v]/d[p_time]]
    float2 du_dv;

    static_assert(
        has_camera_point_to_uvd_jacob_v<CameraType> ||
            has_camera_point_to_uvd_jacob_impl_v<typename CameraType::ImplType>,
        "Camera must implement either ImplType::camera_point_to_uvd_jacob or "
        "CameraType::camera_point_to_uvd_jacob methods to be used within "
        "jac_camera_projection_iteration"
    );
    if constexpr (has_camera_point_to_uvd_jacob_v<CameraType>) {
        camera.camera_point_to_uvd_jacob(
            proj_mode,
            point_cam, jac_point_cam,
            uvd, du_dv
        );
    } else {
        camera.impl.camera_point_to_uvd_jacob(
            point_cam, jac_point_cam,
            uvd, du_dv
        );
    }


    // Time which we used to perform projection
    const float assumed_time = camera.pose_time + p_time;
    // Time which really corresponds to projected UV coordinates
    // real_time = trigger_time + dot(camera.rolling_time, uv);
    const float real_time = camera.pixel_time(float2{uvd.x, uvd.y});
    // Current disagreement
    p_time_residual = assumed_time - real_time;

    // d_residual_dt = d[time_residual] / d[p_time]
    const float d_residual_dt = 1.0f - dot(du_dv, camera.rolling_time);

    // update p_time via Newton-Raphson iteration
    p_time -= p_time_residual / d_residual_dt;
}


// method for cameras with no projection jacobian implementation
template <typename CameraType>
HOSTDEVICE void camera_projection_iteration(
    const CameraType& camera,
    const float3& p_world,
    const float3& p_velocity,
    const ProjectionMode proj_mode,
    float& p_time,
    float3& uvd,
    float& p_time_residual
) {
    // printf("regular iteration\n");
    const auto q_w2c = apply_angular_velocity(camera.q0, camera.w, p_time);
    const float3 p = p_world - camera.t0 + (p_velocity - camera.v) * p_time;
    const float3 p_camera = q_w2c.apply_inv(p);

    uvd = camera.camera_point_to_uvd(proj_mode, p_camera);
    const float assumed_time = camera.pose_time + p_time;
    const float real_time = camera.pixel_time(float2{uvd.x, uvd.y});
    p_time_residual = assumed_time - real_time;
    p_time -= p_time_residual;
}


template <
    uint max_iterations = 5,
    bool force_no_jacobians = false,
    typename CameraType
>
HOSTDEVICE void camera_projection(
    const CameraType& camera,
    const float3 p_world,
    const float3 p_velocity,
    const float residual_thr_early_stop,
    const float max_residual,
    const float max_margin,
    const ProjectionMode proj_mode,
    float4& uvdt,
    bool& proj_valid
) {
    float3 uvd;
    float p_time = 0.0f;
    if (proj_mode == ProjectionMode::NegativeDelta) {
        p_time = -fabsf(camera.rolling_time.x) * 0.5f;
    } else if (proj_mode == ProjectionMode::PositiveDelta) {
        p_time = fabsf(camera.rolling_time.x) * 0.5f;
    }
    float residual = 0.0f;

    #pragma unroll
    for (uint i = 0; i < max_iterations; ++i) {
        if constexpr (
            (has_camera_point_to_uvd_jacob_impl_v<typename CameraType::ImplType> ||
                has_camera_point_to_uvd_jacob_v<CameraType>) && (!force_no_jacobians))
        {
            jac_camera_projection_iteration(
                camera, p_world, p_velocity, proj_mode,
                p_time, uvd, residual);
        } else {
            camera_projection_iteration(
                camera, p_world, p_velocity, proj_mode,
                p_time, uvd, residual);
        }
        if ((uvd.z <= 0.0f) ||
            (uvd.x < -max_margin) || (uvd.x > 1.0f + max_margin) ||
            (uvd.y < -max_margin) || (uvd.y > 1.0f + max_margin))
        {
            proj_valid = false;
            uvdt = {0.0f, 0.0f, 0.0f, 0.0f};
            return;
        }

        if (fabsf(residual) < residual_thr_early_stop) {
            break;
        }
    }

    uvdt = {uvd.x, uvd.y, uvd.z, p_time};
    proj_valid &= (fabsf(residual) < max_residual) & (uvd.z > 0.0f);
}


template <typename CameraType, bool force_no_jacobians = false, uint num_iterations = 5>
std::tuple<at::Tensor, at::Tensor> camera_projection_fwd(
    const at::Tensor& camera_pack,
    const at::Tensor& points,
    const std::optional<at::Tensor>& points_velocity,
    const float residual_thr_early_stop,
    const float max_residual,
    const float max_margin,
    const ProjectionMode proj_mode
) {
    validate_camera_pack<CameraType>(camera_pack);
    const uint N = points.size(0);
    auto out_mask = at::empty(
        {N,},
        camera_pack.options().dtype(at::kBool));
    auto out_uvdt = at::empty({N, 4}, camera_pack.options());

    auto data = std::make_tuple(
        (float4*)out_uvdt.data_ptr(),
        (bool*)out_mask.data_ptr(),
        (const CameraType*)camera_pack.data_ptr(),
        (const float3*)points.data_ptr(),
        points_velocity.has_value() ? (const float3*)points_velocity->data_ptr() : (const float3*)nullptr,
        residual_thr_early_stop, max_residual, max_margin, proj_mode
    );

    INLINE_KERNEL(N, camera_pack.device(), {
        auto [
            out_uvdt_ptr, out_mask_ptr,
            camera_ptr, points_ptr, velocity_ptr,
            solver_thr, max_time_residual, max_margin_val, mode
        ] = data;

        bool cur_mask = true;
        float4 uvdt;
        const auto& cam = camera_ptr[0];
        camera_projection<num_iterations, force_no_jacobians, CameraType>(
            cam,
            points_ptr[idx],
            velocity_ptr ? velocity_ptr[idx]: float3{0.0f, 0.0f, 0.0f},
            solver_thr, max_time_residual,  max_margin_val, mode,uvdt, cur_mask);

        uvdt.w += cam.pose_time;
        out_uvdt_ptr[idx] = uvdt;
        cur_mask &= (uvdt.x >= 0.0f && uvdt.y >= 0.0f);
        cur_mask &= (uvdt.x <= 1.0f && uvdt.y <= 1.0f);
        out_mask_ptr[idx] = cur_mask;
    })

    return {out_uvdt, out_mask};
}


#endif //XSIMGS_PROJ_KERNELS_H
