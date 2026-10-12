#include "gs_masking.h"

#include <toast/utils/vector_math.h>
#include <toast/utils/launch_helpers.h>

#include <ATen/ops/empty.h>


__global__ void to_hwc_rgb_byte_kernel(
    const uint num_pixels,
    const float3* __restrict__ image,
    uchar3* __restrict__ output
) {
    const uint idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_pixels) {return;}

    const float3 src = image[idx];
    unsigned char r = static_cast<unsigned char>(__saturatef(src.x) * 255.0f);
    unsigned char g = static_cast<unsigned char>(__saturatef(src.y) * 255.0f);
    unsigned char b = static_cast<unsigned char>(__saturatef(src.z) * 255.0f);
    output[idx] = {r, g, b};
}

__global__ void to_hwc_rgba_byte_kernel(
    const uint num_pixels,
    const float3* __restrict__ image,
    uchar4* __restrict__ output
) {
    const uint idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_pixels) {return;}

    const float3 src = image[idx];
    unsigned char r = static_cast<unsigned char>(__saturatef(src.x) * 255.0f);
    unsigned char g = static_cast<unsigned char>(__saturatef(src.y) * 255.0f);
    unsigned char b = static_cast<unsigned char>(__saturatef(src.z) * 255.0f);
    output[idx] = {r, g, b, 255};
}

__global__ void to_chw_rgb_byte_kernel(
    const uint num_pixels,
    const float3* __restrict__ image,
    unsigned char* __restrict__ output
) {
    const uint idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_pixels) {return;}

    const float3 src = image[idx];
    output[idx] = static_cast<unsigned char>(__saturatef(src.x) * 255.0f);
    output[num_pixels + idx] = static_cast<unsigned char>(__saturatef(src.y) * 255.0f);
    output[2*num_pixels + idx] = static_cast<unsigned char>(__saturatef(src.z) * 255.0f);
}


std::tuple<at::Tensor, uint, uint> check_image(const at::Tensor& _image) {
    at::Tensor image = _image;
    if (!image.is_cuda()) {
        throw std::runtime_error("image is not a CUDA tensor");
    }
    if (image.dim() != 3) {
        throw std::runtime_error("image must have dimensions [H, W, 3]");
    }
    if (image.dtype() != at::kFloat) {
        throw std::runtime_error("image must have float32 dtype");
    }
    if (image.size(-1) != 3) {
        throw std::runtime_error("image must have dimensions [H, W, 3]");
    }
    if (!image.is_contiguous()) {
        image = image.contiguous();
    }
    return {image, image.size(0), image.size(1)};
}

at::Tensor image_to_hwc_rgb_byte(const at::Tensor& _image) {
    const auto [image, height, width] = check_image(_image);
    const auto result = at::empty(
        {height, width, 3},
        at::TensorOptions().device(image.device()).dtype(at::kByte)
    );
    launch_kernel_cuda(height * width, image.device(), to_hwc_rgb_byte_kernel,
        height * width, (float3*)image.data_ptr(), (uchar3*)result.data_ptr()
    );
    return result;
}

at::Tensor image_to_hwc_rgba_byte(const at::Tensor& _image) {
    const auto [image, height, width] = check_image(_image);
    const auto result = at::empty(
        {height, width, 4},
        at::TensorOptions().device(image.device()).dtype(at::kByte)
    );
    launch_kernel_cuda(height * width, image.device(), to_hwc_rgba_byte_kernel,
        height * width, (float3*)image.data_ptr(), (uchar4*)result.data_ptr()
    );
    return result;
}

at::Tensor image_to_chw_rgb_byte(const at::Tensor& _image) {
    const auto [image, height, width] = check_image(_image);
    const auto result = at::empty(
        {3, height, width},
        at::TensorOptions().device(image.device()).dtype(at::kByte)
    );
    launch_kernel_cuda(height * width, image.device(), to_chw_rgb_byte_kernel,
        height * width, (float3*)image.data_ptr(), (unsigned char*)result.data_ptr()
    );

    return result;
}


void bind_image_functions(py::module_& m) {
    m.def("image_to_hwc_rgb_byte", &image_to_hwc_rgb_byte, py::arg("image"));
    m.def("image_to_hwc_rgba_byte", &image_to_hwc_rgba_byte, py::arg("image"));
    m.def("image_to_chw_rgb_byte", &image_to_chw_rgb_byte, py::arg("image"));
}
