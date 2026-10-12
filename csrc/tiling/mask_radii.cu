#include <torch/csrc/utils/pybind.h>
#include <toast/utils/vector_math.h>
#include <toast/utils/launch_helpers.h>
#include <toast/utils/tensor_helpers.h>


#include "bind_tiling.h"


__global__ void mask_radii_packed_kernel(
    const uint N,
    const uint* __restrict__ packed_mask,
    float2* __restrict__ radii
) {
    const uint idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) {return;}

    const uint pack_val = packed_mask[idx / 32];
    const bool cur_mask_val = (pack_val & (1u << (idx % 32))) > 0;

    if (!cur_mask_val) {
        radii[idx] = {0.0f, 0.0f};
    }
}


void mask_radii_packed(
    const at::Tensor& packed_mask,
    const at::Tensor& radii
) {
    if (radii.size(-1) != 2) {
        throw std::runtime_error("radii must be [..., 2] tensor");
    }
    TORCH_CHECK(packed_mask.dtype() == at::kUInt32, "Packed mask must be UInt32 tensor");

    const uint N = radii.numel() / 2;
    TORCH_CHECK(packed_mask.numel() * 32 >= N, "Packed mask size mismatch with radii");

    check_same_device(packed_mask, radii);
    all_contiguous(packed_mask, radii);

    launch_kernel_cuda(N, radii.device(), mask_radii_packed_kernel,
        N, static_cast<uint *>(packed_mask.data_ptr()),
        static_cast<float2 *>(radii.data_ptr()));
}

__global__ void mask_radii_kernel(
    const uint N,
    const char* __restrict__ mask,
    float2* __restrict__ radii
) {
    const uint idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) {return;}

    if (!mask[idx]) {
        radii[idx] = {0.0f, 0.0f};
    }
}

void mask_radii(
    const at::Tensor& mask,
    const at::Tensor& radii
) {
    if (radii.size(-1) != 2) {
        throw std::runtime_error("radii must be [..., 2] tensor");
    }
    TORCH_CHECK(mask.dtype() == at::kBool, "Mask must be torch.bool tensor");

    const uint N = radii.numel() / 2;
    TORCH_CHECK(mask.numel() == N, "Mask size mismatch with radii");

    check_same_device(mask, radii);
    all_contiguous(mask, radii);

    launch_kernel_cuda(N, radii.device(), mask_radii_kernel,
        N, static_cast<char *>(mask.data_ptr()),
        static_cast<float2 *>(radii.data_ptr()));
}



void bind_mask_radii(py::module_& m) {
    m.def("mask_radii_packed", &mask_radii_packed,
        py::arg("packed_mask"), py::arg("radii"));

    m.def("mask_radii", &mask_radii,
        py::arg("mask"), py::arg("radii"));
}
