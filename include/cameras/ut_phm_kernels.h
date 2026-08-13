#ifndef XSIMGS_UT_PHM_KERNELS_H
#define XSIMGS_UT_PHM_KERNELS_H


#include "ut_kernels.h"


template <
    uint num_iterations = 5,
    bool force_no_jacobians = false,
    typename CameraType
>
__global__ void unscented_transform_phm_kernel(
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

    float2 radii_central = {0.0f, 0.0f},
           radii_left = {0.0f, 0.0f},
           radii_right = {0.0f, 0.0f};
    float2 mean2d_central = {0.0f, 0.0f},
           mean2d_left = {0.0f, 0.0f},
           mean2d_right = {0.0f, 0.0f};
    float depth_central = 0.0f, depth_left = 0.0f, depth_right = 0.0f;
    float u_distance;

    bool result =
       unscented_transform_projection<num_iterations, force_no_jacobians, CameraType>(
           ProjectionMode::Central,
           camera, ut_params, img_size, residual_thr, max_residual, max_margin,
           p_world, p_velocity, p_opacity, deltas,
           mean2d_central, radii_central, depth_central, u_distance
       );

    if (!result) {
        return;
    }

    // todo: check for negative phi_range
    const float u_pi = float(M_PI) * img_size.x / fabsf(camera->impl.phi_range);
    const int result_central = int(result) * (int(u_distance > u_pi) + 1);

    result = unscented_transform_projection<num_iterations, force_no_jacobians, CameraType>(
           ProjectionMode::NegativeDelta,
           camera, ut_params, img_size, residual_thr, max_residual, max_margin,
           p_world, p_velocity, p_opacity, deltas,
           mean2d_left, radii_left, depth_left, u_distance
       );
    const int result_left = int(result) * (int(u_distance > u_pi) + 1);

    result = unscented_transform_projection<num_iterations, force_no_jacobians, CameraType>(
           ProjectionMode::PositiveDelta,
           camera, ut_params, img_size, residual_thr, max_residual, max_margin,
           p_world, p_velocity, p_opacity, deltas,
           mean2d_right, radii_right, depth_right, u_distance
       );
    const int result_right = int(result) * (int(u_distance > u_pi) + 1);

    if (result_left == 1 && result_right == 1) {
        flag[element_idx] = 2;
        radii[element_idx] = radii_left;
        means2d[element_idx] = mean2d_left;
        depths[element_idx] = depth_left;
        return;
    }

    if (result_central == 1) {
        flag[element_idx] = 1;
        radii[element_idx] = radii_central;
        means2d[element_idx] = mean2d_central;
        depths[element_idx] = depth_central;
        return;
    }

    if (result_left != 1 && result_right != 1) {
        return;
    }

    const bool use_left = result_left != 1;
    flag[element_idx] = 1;
    radii[element_idx] = use_left ? radii_left : radii_right;
    means2d[element_idx] = use_left ? mean2d_left : mean2d_right;
    depths[element_idx] = use_left ? depth_left : depth_right;
}


template <
    uint num_iterations = 5,
    bool force_no_jacobians = false,
    typename CameraType
>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>//, torch::Tensor>
launch_unscented_transform_phm(
    const torch::Tensor& camera_pack,
    const torch::Tensor& points,
    const std::optional<torch::Tensor>& point_velocity,
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

    launch_kernel_cuda<CUDA_DEFAULT_NUM_THREADS/2>(
        N, points.device(),
        unscented_transform_phm_kernel<num_iterations, force_no_jacobians, CameraType>,
        N, residual_thr, max_residual, max_margin, ut_values,
        float2{img_width, img_height},
        static_cast<const CameraType *>(camera_pack.data_ptr()),

        static_cast<const float3 *>(points.data_ptr()),
        point_velocity.has_value() ? (float3*)point_velocity->data_ptr() : nullptr,
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


#endif //XSIMGS_UT_PHM_KERNELS_H
