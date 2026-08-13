#include "camera.h"



template <typename CameraType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> launch_camera_ray_fwd(
    const torch::Tensor& camera,
    uint width,
    uint height,
    bool world_space,
    bool normalized,
    bool return_origins,
    bool return_valid
) {
    validate_camera_pack<CameraType>(camera);
    auto ray_o = return_origins ?
        torch::empty({height, width, 3}, camera.options()) : at::Tensor();
    auto ray_d = torch::empty({height, width, 3}, camera.options());
    auto valid = return_valid ?
        torch::empty({height, width}, camera.options().dtype(torch::kBool)) :
        at::Tensor();

    const auto data = std::make_tuple(
        return_origins ? ray_o.data_ptr(): nullptr,
        ray_d.data_ptr(),
        return_valid ? valid.data_ptr() : nullptr,
        static_cast<CameraType*>(camera.data_ptr()),
        uint2{width, height}, world_space, normalized);

    INLINE_KERNEL(height * width, camera.device(), {
        const auto [
            ray_o_ptr, ray_d_ptr, valid_ptr, camera_ptr,
            size, in_world_space, normalize] = data;
        const auto px = camera_ptr->pixel_idx(idx, size);

        float3 ray_o_val{0.0f, 0.0f, 0.0f};
        float3 ray_d_val;
        if (in_world_space) {
            const auto ray = camera_ptr->camera_ray(px, size, normalize);
            ray_o_val = std::get<0>(ray);
            ray_d_val = std::get<1>(ray);
        } else {
            ray_d_val = camera_ptr->camera_ray_cs(px, size, normalize);
        }

        if (ray_o_ptr) {
            static_cast<float3 *>(ray_o_ptr)[idx] = ray_o_val;
        }
        static_cast<float3 *>(ray_d_ptr)[idx] = ray_d_val;
        if (valid_ptr) {
            static_cast<bool *>(valid_ptr)[idx] =
                camera_ptr->ray_valid(camera_ptr->pixel_to_uv(px, size));
        }
    })

    return {ray_o, ray_d, valid};
}

template <typename CameraType, uint num_threads = CUDA_DEFAULT_NUM_THREADS>
__global__ void camera_ray_backward_kernel_cuda(
    uint width,
    uint height,
    bool world_space,
    bool normalized,
    const CameraType* camera,
    const float3* d_ray_o,
    const float3* d_ray_d,
    char* block_outputs
) {
    const uint pixel_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (pixel_idx >= width * height) return;

    const uint2 size{width, height};
    const int2 px{int(pixel_idx % size.x), int(pixel_idx / size.x)};
    const float2 uv = camera->pixel_to_uv(px, size);

    __shared__ BlockReduceType<num_threads> sh_mem;

    float3 ray, d_ray;
    if (world_space) {
        const float time_delta = camera->pixel_time_delta(uv);
        // t = t0 + delta_time * v
        const float3 d_t = d_ray_o[pixel_idx];
        block_reduce_float3(block_outputs, offsetof(CameraType, t0), d_t)
        block_reduce_float3(block_outputs, offsetof(CameraType, v), time_delta * d_t)

        float d_time_delta = dot(d_t, camera->v);
        ray = camera->impl.pixel_to_camera_ray(uv);
        const float3 norm_ray = normalized ? normalize(ray): ray;

        const auto q0_std = camera->q0.std();
        const auto q = apply_angular_velocity(q0_std, camera->w, time_delta);
        const auto [d_q, d_norm_ray] = quatf::apply_backward(q, norm_ray, d_ray_d[pixel_idx]);

        const auto [dq0_std, dw, d_time_delta2] = apply_angular_velocity_backward(q0_std, camera->w, time_delta, d_q);
        d_time_delta += d_time_delta2;

        block_reduce_float3(block_outputs, offsetof(CameraType, w), dw)
        block_reduce_float4(block_outputs, offsetof(CameraType, q0), quatf::std_backward(camera->q0, dq0_std))

        // time_delta = trigger_time + dot(uv, rolling_time) - pose_time
        block_reduce_float(block_outputs, offsetof(CameraType, trigger_time), d_time_delta)
        block_reduce_float(block_outputs, offsetof(CameraType, pose_time), -d_time_delta)
        block_reduce_float2(block_outputs, offsetof(CameraType, rolling_time), d_time_delta * uv)

        d_ray = d_norm_ray;
    } else {
        ray = camera->camera_ray_cs(uv, normalized);
        d_ray = d_ray_d[pixel_idx];
    }

    d_ray = normalized ? normalize_bwd<Float32>(ray, d_ray): d_ray;
    camera->impl.pixel_to_camera_ray_backward(uv, d_ray, block_outputs, sh_mem, offsetof(CameraType, impl));
}



template <typename CameraType>
static torch::Tensor launch_camera_ray_bwd(
    const torch::Tensor& packed_camera,
    const torch::Tensor& d_ray_o,
    const torch::Tensor& d_ray_d,
    bool world_space,
    bool normalized)
{
    if (!packed_camera.device().is_cuda()) {
        throw std::runtime_error("camera_ray backward kernel is implemented only for CUDA");
    }
    uint height = d_ray_d.size(0);
    uint width = d_ray_d.size(1);
    uint num_pixels = height * width;

    auto out = torch::zeros({
        CameraType::pack_elements(),
        (num_pixels + CUDA_DEFAULT_NUM_THREADS - 1) / CUDA_DEFAULT_NUM_THREADS
    }, d_ray_d.options());

    launch_kernel_cuda<CUDA_DEFAULT_NUM_THREADS>(
        num_pixels, d_ray_d.device(),
        camera_ray_backward_kernel_cuda<CameraType>,
        width, height, world_space, normalized,
        (CameraType*)packed_camera.data_ptr(),
        (float3*)d_ray_o.data_ptr(), 
        (float3*)d_ray_d.data_ptr(), 
        (char*)out.data_ptr()
    );

    return out.sum(-1).view({1, -1});
}


template <typename CameraType>
class CameraRaysFunction: public torch::autograd::Function<CameraRaysFunction<CameraType>> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor packed_camera,
        uint width,
        uint height,
        bool world_space,
        bool normalized,
        bool return_origins,
        bool return_valid
    ) {
        ctx->save_for_backward({packed_camera});
        ctx->saved_data["world_space"] = world_space;
        ctx->saved_data["normalized"] = normalized;
        ctx->saved_data["return_origins"] = return_origins;
        auto rays = launch_camera_ray_fwd<CameraType>(
            packed_camera, width, height,
            world_space, normalized, return_origins, return_valid);
        torch::Tensor ray_o = std::get<0>(rays);
        torch::Tensor ray_d = std::get<1>(rays);
        torch::Tensor valid = std::get<2>(rays);
        if (return_valid) {
            ctx->mark_non_differentiable(torch::autograd::variable_list{valid});
            return {ray_o, ray_d, valid};
        }
        return {ray_o, ray_d};
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        const auto packed_camera = ctx->get_saved_variables()[0];
        return {launch_camera_ray_bwd<CameraType>(
            packed_camera,
            grad_outputs[0],
            grad_outputs[1],
            ctx->saved_data["world_space"].toBool(),
            ctx->saved_data["normalized"].toBool()
        ), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
           torch::Tensor(), torch::Tensor()};
    }
};

template <typename CameraType>
auto camera_rays_function(
    torch::Tensor packed_camera,
    uint width,
    uint height,
    bool world_space,
    bool normalized,
    bool return_origins,
    bool return_valid
) {
    return CameraRaysFunction<CameraType>::apply(
        packed_camera, width, height, world_space, normalized, return_origins,
        return_valid);
}

