#ifndef XSIMGS_UT_KERNELS_H
#define XSIMGS_UT_KERNELS_H

#include <toast/math/quat.h>

#include "projection_kernels.h"


#define ALPHA_THR_MIN (1.0f/255.0f)


inline float4 ut_weights(
    const float ut_alpha = 0.1f,
    const float ut_beta = 2.0f,
    const float ut_kappa = 0.0f
) {
    const float ut_lambda = ut_alpha * ut_alpha * (3 + ut_kappa) - 3;

    const float ut_weight_mean_0 = ut_lambda / (3 + ut_lambda);
    const float ut_weight_cov_0 =
        ut_lambda / (3 + ut_lambda) + (1 - ut_alpha * ut_alpha + ut_beta);
    const float ut_weight_other = 1 / (2 * (3 + ut_lambda));
    const float sqrt_D_lambda = std::sqrt(ut_lambda + 3);

    return {
        ut_weight_mean_0,
        ut_weight_cov_0,
        ut_weight_other,
        sqrt_D_lambda
    };
}

HOSTDEVICE void unscented_transform_deltas(
    const float sqrt_D_lambda,
    const quatf& p_quat,
    const float3& p_scale,
    float3* deltas
) {
    // conjugate since we need matrix columns, not rows in deltas
    p_quat.conjugate().to_matrix(deltas);
    deltas[0] *= p_scale.x * sqrt_D_lambda;
    deltas[1] *= p_scale.y * sqrt_D_lambda;
    deltas[2] *= p_scale.z * sqrt_D_lambda;
}


template <
    uint num_iterations = 5,
    bool force_no_jacobians = false,
    typename CameraType
>
HOSTDEVICE bool unscented_transform(
    const CameraType& camera,
    const float4& ut_weights,
    const float2& img_size,
    const ProjectionMode proj_mode,
    const float residual_thr,
    const float max_residual,
    const float max_margin,

    const float3 p_world,
    const float3 p_velocity,
    const float3* deltas,

    float2& mean2d,
    float& depth,
    float& u_distance,
    float3& cov2d
) {
    float4 uvdt; bool is_valid = true;
    float2 points[7];

    camera_projection<num_iterations, force_no_jacobians>(
        camera, p_world, p_velocity,
        residual_thr, max_residual, max_margin, proj_mode,
        uvdt, is_valid
    );
    if (!is_valid) {
        return false;
    }

    const auto& [
        weight_avg0,
        weight_cov0,
        weight,
        sqrt_D_lambda
    ] = ut_weights;

    points[0] = float2{uvdt.x, uvdt.y} * img_size;
    mean2d = weight_avg0 * points[0];
    depth = uvdt.z;
    float u_min = points[0].x, u_max = points[0].x;

    float sign = 1.0f;
    #pragma unroll
    for (uint i = 0; i < 6; ++i) {
        camera_projection<num_iterations, force_no_jacobians>(
            camera, p_world + sign * deltas[i / 2], p_velocity,
            residual_thr, max_residual, max_margin, proj_mode,
            uvdt, is_valid
        );
        points[i+1] = float2{uvdt.x, uvdt.y} * img_size;
        mean2d += weight * points[i+1];
        u_min = fminf(u_min, points[i+1].x);
        u_max = fmaxf(u_max, points[i+1].x);
        sign *= -1.0f;
    }

    // TODO: which depth use for sorting? perspective or ray length?

    if (!is_valid) {
        return false;
    }
    u_distance = u_max - u_min;
    // combine covariance from statistics
    float2 vec = points[0] - mean2d;
    cov2d = weight_cov0 * make_float3(vec.x * vec.x, vec.x * vec.y, vec.y * vec.y);
    #pragma unroll
    for (int i = 1; i < 7; ++i) {
        vec = points[i] - mean2d;
        cov2d += weight * make_float3(vec.x * vec.x, vec.x * vec.y, vec.y * vec.y);
    }

    // TODO: should we add eps2d blur?
    const float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;

    if (det <= 0.0f || cov2d.x < 0 || cov2d.z < 0) {
        return false;
    }

    return true;
}


template <
    uint num_iterations = 5,
    bool force_no_jacobians = false,
    typename CameraType
>
HOSTDEVICE bool unscented_transform_projection(
    ProjectionMode mode,
    const CameraType* camera,
    const float4& ut_params,
    const float2& img_size,
    float residual_thr,
    float max_residual,
    float max_margin,

    const float3& p_world,
    const float3& p_velocity,
    const float& p_opacity,
    const float3* deltas,

    float2& mean2d,
    float2& radii,
    float& depth,
    float& u_distance
) {
    float3 cov2d;

    const bool result = unscented_transform<num_iterations, force_no_jacobians, CameraType>(
        *camera, ut_params, img_size,
        mode, residual_thr, max_residual, max_margin,
        p_world, p_velocity, deltas,
        mean2d, depth, u_distance, cov2d);

    const float extend = fminf(
        3.33f, sqrtf(2.0f * logf(p_opacity / ALPHA_THR_MIN)));
    radii = {
        extend * sqrtf(fmaxf(0.0f, cov2d.x)),
        extend * sqrtf(fmaxf(0.0f, cov2d.z))
    };
    const bool intersects_x = !(
        ((mean2d.x + radii.x) < 0) || (img_size.x < (mean2d.x - radii.x)));
    const bool intersects_y = !(
        ((mean2d.y + radii.y) < 0) || (img_size.y < (mean2d.y - radii.y)));

    return result && intersects_x && intersects_y;
}


template <
    uint num_iterations = 5,
    bool force_no_jacobians = false,
    typename CameraType
>
__global__ void unscented_transform_kernel(
    const uint N,
    const float residual_thr,
    const float max_residual,
    const float max_margin,
    const float4 ut_params,
    const float2 img_size,

    const CameraType* camera,
    const float3* __restrict__ points,
    const float3* __restrict__ point_velocities,
    const float3* __restrict__ scales,
    const quatf* __restrict__ quats,
    const float* __restrict__ opacity,

    char* __restrict__ flag,
    float2* __restrict__ radii,
    float2* __restrict__ means2d,
    float* __restrict__ depths
) {
    const uint element_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (element_idx >= N) {return;}

    const float3 p_world = points[element_idx];
    const float3 p_velocity = point_velocities ? point_velocities[element_idx] : float3{0.0f, 0.0f, 0.0f};
    const float3 p_scale = scales[element_idx];
    const quatf p_quat = quats[element_idx];
    const float p_opacity = opacity ? opacity[element_idx] : 1.0f;

    if (p_opacity < ALPHA_THR_MIN) {
        return;
    }

    float3 deltas[3];
    unscented_transform_deltas(ut_params.w, p_quat, p_scale, deltas);

    float2 cur_radii;
    float2 mean2d;
    float depth;
    float u_distance;

    const bool result =
        unscented_transform_projection<num_iterations, force_no_jacobians, CameraType>(
            ProjectionMode::Central,
            camera, ut_params, img_size, residual_thr, max_residual, max_margin,
            p_world, p_velocity, p_opacity, deltas,
            mean2d, cur_radii, depth, u_distance
        );

    if (!result) return;
    radii[element_idx] = cur_radii;
    flag[element_idx] = true;
    means2d[element_idx] = mean2d;
    depths[element_idx] = depth;
}


template <
    uint num_iterations = 5,
    bool force_no_jacobians = false,
    typename CameraType
>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>//, torch::Tensor>
launch_unscented_transform(
    const torch::Tensor& camera_pack,
    const torch::Tensor& points,
    const std::optional<torch::Tensor>& point_velocities,
    const torch::Tensor& scales,
    const torch::Tensor& quats,
    const std::optional<torch::Tensor>& opacity,
    float img_width,
    float img_height,

    float ut_alpha,
    float ut_beta,
    float ut_kappa,
    float residual_thr,
    float max_residual,
    float max_margin
) {
    validate_camera_pack<CameraType>(camera_pack);
    check_same_device(camera_pack, points, scales, quats);
    const uint N = points.size(0);

    auto flags = torch::zeros({N,},
        at::TensorOptions(torch::kByte).device(points.device()));
    auto radii = torch::zeros({N, 2}, points.options());
    auto mean2d = torch::zeros({N, 2}, points.options());
    auto depths = torch::zeros({N, }, points.options());
    // auto cov2d = torch::zeros({N, 3}, points.options());

    float4 ut_values = ut_weights(ut_alpha, ut_beta, ut_kappa);

    launch_kernel_cuda(
        N, points.device(),
        unscented_transform_kernel<num_iterations, force_no_jacobians, CameraType>,
        N, residual_thr, max_residual, max_margin, ut_values,
        float2{img_width, img_height},
        static_cast<const CameraType *>(camera_pack.data_ptr()),

        static_cast<const float3 *>(points.data_ptr()),
        point_velocities.has_value() ? static_cast<const float3 *>(point_velocities->data_ptr()) : nullptr,
        static_cast<const float3 *>(scales.data_ptr()),
        static_cast<const quatf *>(quats.data_ptr()),
        opacity.has_value() ? (float*)opacity->data_ptr() : nullptr,

        static_cast<char*>(flags.data_ptr()),
        static_cast<float2 *>(radii.data_ptr()),
        static_cast<float2 *>(mean2d.data_ptr()),
        static_cast<float *>(depths.data_ptr())
    );

    return {flags, radii, mean2d, depths};
}


#endif //XSIMGS_UT_KERNELS_H
