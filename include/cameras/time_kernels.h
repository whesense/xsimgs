#include "camera.h"
#include <torch/extension.h>


template <typename CameraType>
torch::Tensor pixel_time_fwd(
    const torch::Tensor& camera_pack,
    const uint width,
    const uint height
) {
    validate_camera_pack<CameraType>(camera_pack);
    auto out = torch::empty({height, width}, camera_pack.options());

    const auto data = std::make_tuple(
        (float*)out.data_ptr(),
        (CameraType*)camera_pack.data_ptr(),
        uint2{width, height}
    );

    INLINE_KERNEL(height * width, camera_pack.device(), {
        const auto [out_ptr, camera_ptr, size] = data;
        const auto px = camera_ptr->pixel_idx(idx, size);
        const auto uv = camera_ptr->pixel_to_uv(px, size);
        out_ptr[idx] = camera_ptr->pixel_time(uv);
    })

    return out;
}

template <typename CameraType>
torch::Tensor pixel_time_from_uv_fwd(
    const torch::Tensor& camera_pack,
    const torch::Tensor& uv
) {
    validate_camera_pack<CameraType>(camera_pack);
    TORCH_CHECK(uv.size(-1) == 2, "UV coordinates last dimension must be 2")
    TORCH_CHECK(uv.device() == camera_pack.device(), "Camera and UV-s must be on the same device")
    TORCH_CHECK(uv.dtype() == torch::kFloat32, "UV coordinates must be float32 tensor")
    const uint N = uv.numel() / 2;

    auto sizes = uv.sizes().vec();
    sizes.pop_back();
    auto out = torch::empty(sizes, uv.options());

    const auto data = std::make_tuple(
        (float*)out.data_ptr(),
        (CameraType*)camera_pack.data_ptr(),
        (float2*)uv.data_ptr()
    );

    INLINE_KERNEL(N, camera_pack.device(), {
        const auto [out_ptr, camera_ptr, uv_ptr] = data;
        out_ptr[idx] = camera_ptr->pixel_time(uv_ptr[idx]);
    })

    return out;
}

template <typename CameraType>
torch::Tensor pixel_time_delta_fwd(
    const torch::Tensor& camera_pack,
    const uint width,
    const uint height
) {
    validate_camera_pack<CameraType>(camera_pack);
    auto out = torch::empty({height, width}, camera_pack.options());

    const auto data = std::make_tuple(
        (float*)out.data_ptr(),
        (CameraType*)camera_pack.data_ptr(),
        uint2{width, height}
    );

    INLINE_KERNEL(height * width, camera_pack.device(), {
        const auto [out_ptr, camera_ptr, size] = data;
        const auto px = camera_ptr->pixel_idx(idx, size);
        const auto uv = camera_ptr->pixel_to_uv(px, size);
        out_ptr[idx] = camera_ptr->pixel_time_delta(uv);
    })

    return out;
}

template <typename CameraType>
torch::Tensor pixel_time_delta_from_uv_fwd(
    const torch::Tensor& camera_pack,
    const torch::Tensor& uv
) {
    validate_camera_pack<CameraType>(camera_pack);
    TORCH_CHECK(uv.device() == camera_pack.device(), "Camera and UV-s must be on the same device")
    TORCH_CHECK(uv.size(-1) == 2, "UV coordinates last dimension must be 2")
    TORCH_CHECK(uv.device() == camera_pack.device(), "Camera and UV-s must be on the same device")
    TORCH_CHECK(uv.dtype() == torch::kFloat32, "UV coordinates must be float32 tensor")
    const uint N = uv.numel() / 2;

    auto sizes = uv.sizes().vec();
    sizes.pop_back();
    auto out = torch::empty(sizes, uv.options());

    const auto data = std::make_tuple(
        (float*)out.data_ptr(),
        (CameraType*)camera_pack.data_ptr(),
        (float2*)uv.data_ptr()
    );

    INLINE_KERNEL(N, camera_pack.device(), {
        const auto [out_ptr, camera_ptr, uv_ptr] = data;
        out_ptr[idx] = camera_ptr->pixel_time_delta(uv_ptr[idx]);
    })

    return out;
}


template <typename CameraType, uint num_threads = CUDA_DEFAULT_NUM_THREADS>
__global__ void pixel_time_bwd_kernel(
    uint width,
    uint height,
    const CameraType* __restrict__ camera,
    const float* __restrict__ grad,
    char* __restrict__ block_outputs
) {
    const uint pixel_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (pixel_idx >= width * height) return;

    const uint2 size{width, height};
    const int2 px{int(pixel_idx % size.x), int(pixel_idx / size.x)};
    const float2 uv = camera->pixel_to_uv(px, size);

    const float cur_grad = grad[pixel_idx];

    __shared__ union {
        typename cub::BlockReduce<float2, num_threads>::TempStorage float2_storage;
        typename cub::BlockReduce<float, num_threads>::TempStorage float_storage;
    } sh_mem;

    block_reduce_float(block_outputs, offsetof(CameraType, trigger_time), cur_grad)
    // block_reduce_float(block_outputs, offsetof(CameraType, pose_time), -d_time_delta)
    block_reduce_float2(block_outputs, offsetof(CameraType, rolling_time), cur_grad * uv)
}



template <typename CameraType, uint num_threads = CUDA_DEFAULT_NUM_THREADS>
__global__ void pixel_time_from_uv_bwd_kernel(
    uint N,
    const CameraType* __restrict__ camera,
    const float2* __restrict__ uvs,
    const float* __restrict__ grad,
    char* __restrict__ block_outputs,
    float2* __restrict__ d_uvs
) {
    const uint element_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (element_idx >= N) return;

    const float2 uv = uvs[element_idx];
    const float cur_grad = grad[element_idx];

    __shared__ union {
        typename cub::BlockReduce<float2, num_threads>::TempStorage float2_storage;
        typename cub::BlockReduce<float, num_threads>::TempStorage float_storage;
    } sh_mem;

    d_uvs[element_idx] = cur_grad * camera->rolling_time;
    block_reduce_float(block_outputs, offsetof(CameraType, trigger_time), cur_grad)
    // block_reduce_float(block_outputs, offsetof(CameraType, pose_time), -d_time_delta)
    block_reduce_float2(block_outputs, offsetof(CameraType, rolling_time), cur_grad * uv)
}


template <typename CameraType>
torch::Tensor pixel_time_bwd(
    const torch::Tensor& camera_pack,
    const torch::Tensor& grad
) {
    if (!camera_pack.device().is_cuda()) {
        throw std::runtime_error("pixel_time_bwd kernel is implemented only for CUDA");
    }
    uint height = grad.size(0);
    uint width = grad.size(1);
    uint num_pixels = height * width;

    auto out = torch::zeros({
        CameraType::pack_elements(),
        (num_pixels + CUDA_DEFAULT_NUM_THREADS - 1) / CUDA_DEFAULT_NUM_THREADS
    }, grad.options());

    launch_kernel_cuda<CUDA_DEFAULT_NUM_THREADS>(
        num_pixels, grad.device(), 
        pixel_time_bwd_kernel<CameraType>,
        width, height, 
        (CameraType*)camera_pack.data_ptr(),
        (float*)grad.data_ptr(), 
        (char*)out.data_ptr()
    );

    return out.sum(-1).view({1, -1});
}

template <typename CameraType>
std::tuple<torch::Tensor, torch::Tensor> pixel_time_from_uv_bwd(
    const torch::Tensor& camera_pack,
    const torch::Tensor& uvs,
    const torch::Tensor& grad
) {
    if (!camera_pack.device().is_cuda()) {
        throw std::runtime_error("pixel_time_bwd kernel is implemented only for CUDA");
    }
    uint N = uvs.numel() / 2;

    auto d_camera = torch::zeros({
        CameraType::pack_elements(),
        (N + CUDA_DEFAULT_NUM_THREADS - 1) / CUDA_DEFAULT_NUM_THREADS
    }, grad.options());

    auto d_uvs = torch::empty_like(uvs);

    launch_kernel_cuda<CUDA_DEFAULT_NUM_THREADS>(
        N, grad.device(),
        pixel_time_from_uv_bwd_kernel<CameraType>,
        N,
        (const CameraType*)camera_pack.data_ptr(),
        (const float2*)uvs.data_ptr(),
        (const float*)grad.data_ptr(),
        (char*)d_camera.data_ptr(),
        (float2*)d_uvs.data_ptr()
    );

    return std::tuple{d_camera.sum(-1).view({1, -1}), d_uvs};
}


template <typename CameraType, uint num_threads = CUDA_DEFAULT_NUM_THREADS>
__global__ void pixel_time_delta_bwd_kernel(
    uint width,
    uint height,
    const CameraType* camera,
    const float* grad,
    char* block_outputs
) {
    const uint pixel_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (pixel_idx >= width * height) return;

    const uint2 size{width, height};
    const int2 px{int(pixel_idx % size.x), int(pixel_idx / size.x)};
    const float2 uv = camera->pixel_to_uv(px, size);

    const float cur_grad = grad[pixel_idx];

    __shared__ union {
        typename cub::BlockReduce<float2, num_threads>::TempStorage float2_storage;
        typename cub::BlockReduce<float, num_threads>::TempStorage float_storage;
    } sh_mem;

    block_reduce_float(block_outputs, offsetof(CameraType, trigger_time), cur_grad)
    block_reduce_float(block_outputs, offsetof(CameraType, pose_time), -cur_grad)
    block_reduce_float2(block_outputs, offsetof(CameraType, rolling_time), cur_grad * uv)
}

template <typename CameraType, uint num_threads = CUDA_DEFAULT_NUM_THREADS>
__global__ void pixel_time_delta_from_uv_bwd_kernel(
    uint N,
    const CameraType* __restrict__ camera,
    const float2* __restrict__ uvs,
    const float* __restrict__ grad,
    char* __restrict__ block_outputs,
    float2* __restrict__ d_uvs
) {
    const uint element_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (element_idx >= N) return;

    const float2 uv = uvs[element_idx];
    const float cur_grad = grad[element_idx];

    __shared__ union {
        typename cub::BlockReduce<float2, num_threads>::TempStorage float2_storage;
        typename cub::BlockReduce<float, num_threads>::TempStorage float_storage;
    } sh_mem;

    d_uvs[element_idx] = cur_grad * camera->rolling_time;
    block_reduce_float(block_outputs, offsetof(CameraType, trigger_time), cur_grad)
    block_reduce_float(block_outputs, offsetof(CameraType, pose_time), -cur_grad)
    block_reduce_float2(block_outputs, offsetof(CameraType, rolling_time), cur_grad * uv)
}


template <typename CameraType>
torch::Tensor pixel_time_delta_bwd(
    const torch::Tensor& camera_pack,
    const torch::Tensor& grad
) {
    if (!camera_pack.device().is_cuda()) {
        throw std::runtime_error("pixel_time_bwd kernel is implemented only for CUDA");
    }
    uint height = grad.size(0);
    uint width = grad.size(1);
    uint num_pixels = height * width;

    auto out = torch::zeros({
        CameraType::pack_elements(),
        (num_pixels + CUDA_DEFAULT_NUM_THREADS - 1) / CUDA_DEFAULT_NUM_THREADS
    }, grad.options());

    launch_kernel_cuda<CUDA_DEFAULT_NUM_THREADS>(
        num_pixels, grad.device(), 
        pixel_time_delta_bwd_kernel<CameraType>,
        width, height, 
        (CameraType*)camera_pack.data_ptr(),
        (float*)grad.data_ptr(), 
        (char*)out.data_ptr()
    );

    return out.sum(-1).view({1, -1});
}


template <typename CameraType>
std::tuple<torch::Tensor, torch::Tensor> pixel_time_delta_from_uv_bwd(
    const torch::Tensor& camera_pack,
    const torch::Tensor& uvs,
    const torch::Tensor& grad
) {
    if (!camera_pack.device().is_cuda()) {
        throw std::runtime_error("pixel_time_bwd kernel is implemented only for CUDA");
    }
    uint N = uvs.numel() / 2;

    auto d_camera = torch::zeros({
        CameraType::pack_elements(),
        (N + CUDA_DEFAULT_NUM_THREADS - 1) / CUDA_DEFAULT_NUM_THREADS
    }, grad.options());

    auto d_uvs = torch::empty_like(uvs);

    launch_kernel_cuda<CUDA_DEFAULT_NUM_THREADS>(
        N, grad.device(),
        pixel_time_delta_from_uv_bwd_kernel<CameraType>,
        N,
        (const CameraType*)camera_pack.data_ptr(),
        (const float2*)uvs.data_ptr(),
        (const float*)grad.data_ptr(),
        (char*)d_camera.data_ptr(),
        (float2*)d_uvs.data_ptr()
    );

    return std::tuple{d_camera.sum(-1).view({1, -1}), d_uvs};
}




template <typename CameraType>
class CameraPixelTimeFunction: public torch::autograd::Function<CameraPixelTimeFunction<CameraType>> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor packed_camera,
        uint width,
        uint height
    ) {
        ctx->save_for_backward({packed_camera});
        return pixel_time_fwd<CameraType>(packed_camera, width, height);
    }

    static torch::autograd::tensor_list backward(
        const torch::autograd::AutogradContext *ctx,
        const torch::autograd::tensor_list& grad_outputs
    ) {
        const auto packed_camera = ctx->get_saved_variables()[0];
        return {
            pixel_time_bwd<CameraType>(packed_camera, grad_outputs[0].contiguous()),
            torch::Tensor(),
            torch::Tensor()
        };
    }
};


template <typename CameraType>
auto camera_pixel_time(const torch::Tensor& pack, uint width, uint height) {
    return CameraPixelTimeFunction<CameraType>::apply(pack, width, height);
}



template <typename CameraType>
class CameraPixelTimeFromUVFunction: public torch::autograd::Function<CameraPixelTimeFromUVFunction<CameraType>> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor packed_camera,
        torch::Tensor uvs
    ) {
        ctx->save_for_backward({packed_camera, uvs});
        return pixel_time_from_uv_fwd<CameraType>(packed_camera, uvs);
    }

    static torch::autograd::tensor_list backward(
        const torch::autograd::AutogradContext *ctx,
        const torch::autograd::tensor_list& grad_outputs
    ) {
        const auto packed_camera = ctx->get_saved_variables()[0];
        const auto uvs = ctx->get_saved_variables()[1];
        const auto [d_camera, d_uv] = pixel_time_from_uv_bwd<CameraType>(packed_camera, uvs, grad_outputs[0].contiguous());
        return {d_camera, d_uv};
    }
};


template <typename CameraType>
auto camera_pixel_time_from_uv(const torch::Tensor& pack, const torch::Tensor& uvs) {
    return CameraPixelTimeFromUVFunction<CameraType>::apply(pack, uvs);
}


template <typename CameraType>
class CameraPixelTimeDeltaFunction: public torch::autograd::Function<CameraPixelTimeDeltaFunction<CameraType>> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor packed_camera,
        uint width,
        uint height
    ) {
        ctx->save_for_backward({packed_camera});
        return pixel_time_delta_fwd<CameraType>(packed_camera, width, height);
    }

    static torch::autograd::tensor_list backward(
        const torch::autograd::AutogradContext *ctx,
        const torch::autograd::tensor_list& grad_outputs
    ) {
        const auto packed_camera = ctx->get_saved_variables()[0];
        return {
            pixel_time_delta_bwd<CameraType>(packed_camera, grad_outputs[0].contiguous()),
            torch::Tensor(),
            torch::Tensor()
        };
    }
};


template <typename CameraType>
auto camera_pixel_time_delta(const torch::Tensor& pack, uint width, uint height) {
    return CameraPixelTimeDeltaFunction<CameraType>::apply(pack, width, height);
}



template <typename CameraType>
class CameraPixelTimeDeltaFromUVFunction: public torch::autograd::Function<CameraPixelTimeDeltaFromUVFunction<CameraType>> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor packed_camera,
        torch::Tensor uvs
    ) {
        ctx->save_for_backward({packed_camera, uvs});
        return pixel_time_delta_from_uv_fwd<CameraType>(packed_camera, uvs);
    }

    static torch::autograd::tensor_list backward(
        const torch::autograd::AutogradContext *ctx,
        const torch::autograd::tensor_list& grad_outputs
    ) {
        const auto packed_camera = ctx->get_saved_variables()[0];
        const auto uvs = ctx->get_saved_variables()[1];
        const auto [d_camera, d_uv] = pixel_time_delta_from_uv_bwd<CameraType>(packed_camera, uvs, grad_outputs[0].contiguous());
        return {d_camera, d_uv};
    }
};


template <typename CameraType>
auto camera_pixel_time_delta_from_uv(const torch::Tensor& pack, const torch::Tensor& uvs) {
    return CameraPixelTimeDeltaFromUVFunction<CameraType>::apply(pack, uvs);
}
