#include "gs_masking.h"

#include <torch/extension.h>
#include <toast/utils/launch_helpers.h>
#include <toast/utils/tensor_helpers.h>


__global__ void mask_from_instances_kernel(
     const uint N,
     const int* __restrict__ instance_ids,
     const char* __restrict__ instance_mask,
     uint* __restrict__ outputs
) {
    const uint idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) {return;}

    const int instance_id = instance_ids[idx];
    const int value = (instance_id >= 0) ? static_cast<int>(instance_mask[instance_id]) : 1;
    const uint result = __ballot_sync(0xFFFFFFFF, value);

    if ((threadIdx.x % 32) == 0) {
        outputs[idx / 32] = result;
    }
}


torch::Tensor mask_from_instances(
     const torch::Tensor& instance_ids,
     const torch::Tensor& instance_mask
) {
    check_same_device(instance_ids, instance_mask);
    all_contiguous(instance_ids, instance_mask);

    const uint N = instance_ids.numel();
    torch::Tensor output = torch::empty({(N + 31) / 32},
         at::TensorOptions().dtype(torch::kUInt32).device(instance_ids.device()));

    launch_kernel_cuda(N, instance_ids.device(), mask_from_instances_kernel,
        N, (int*)instance_ids.data_ptr(), (char*)instance_mask.data_ptr(),
        (uint*)output.data_ptr());

    return output;
}

void bind_gs_masking_functions(py::module_& m) {
    m.def("packed_mask_from_instances", &mask_from_instances,
        py::arg("instance_ids"), py::arg("instance_mask"));
}
