#include "smpl.h"

#include <toast/math/se3.h>
#include <toast/utils/launch_helpers.h>
#include <toast/utils/tensor_helpers.h>

#include <c10/cuda/CUDAGuard.h>

#include <utils/reduce_helpers.cuh>
#include "utils.cuh"




template <int num_threads = 1024>
__global__ void smpl_ids_compaction_kernel(
    uint num_smpl_instances,
    const uint* __restrict__ smpl_instance_ids, // [num_smpl_insatnces,]
    const unsigned char* __restrict__ global_instance_mask, // [K]
    uint* __restrict__ output_ids,
    uint* __restrict__ cnt
) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint is_valid = 0;
    if (idx < num_smpl_instances) {
        // Do not quit invalid threads,
        // otherwise subsequent reduces and scans
        // would result in undefined behavior
        is_valid = global_instance_mask[smpl_instance_ids[idx]];
    }

    using BlockReduce = cub::BlockReduce<uint, num_threads>;
    using BlockScan = cub::BlockScan<uint, num_threads>;

    __shared__ union {
        typename BlockReduce::TempStorage sum_storage;
        typename BlockScan::TempStorage prefix_sum_storage;
    } sh_mem;

    uint total_cnt = BlockReduce(sh_mem.sum_storage).Sum(is_valid);

    uint output_id;
    __syncthreads(); // barrier for shared memory usage
    BlockScan(sh_mem.prefix_sum_storage).ExclusiveSum(is_valid, output_id);

    if (is_valid) {
        output_ids[output_id] = idx;
    }
    if (threadIdx.x == 0) {
        cnt[0] = total_cnt;
    }
}

torch::Tensor smpl_ids_compaction(
    const torch::Tensor& instance_ids,
    const torch::Tensor& instance_mask
) {
    const at::cuda::OptionalCUDAGuard device_guard(instance_ids.device());

    torch::Tensor cur_ids = torch::empty(
        {instance_ids.numel()}, instance_ids.options());
    torch::Tensor out_cnt = torch::empty({1}, instance_ids.options());

    constexpr uint num_threads = 1024;
    const uint num_blocks = (instance_ids.numel() + num_threads - 1) / num_threads;
    if (num_blocks > 1) {
        throw std::runtime_error("Only up to 1024 instances supported");
    }
    const auto& stream = at::cuda::getCurrentCUDAStream();

    smpl_ids_compaction_kernel<num_threads><<<num_blocks, num_threads, 0, stream>>>(
        instance_ids.numel(),
        (uint*)instance_ids.data_ptr(),
        (unsigned char*)instance_mask.data_ptr(),
        (uint*)cur_ids.data_ptr(),
        (uint*)out_cnt.data_ptr()
    );
    cudaStreamSynchronize(stream);
    uint num_values;
    cudaMemcpy(&num_values, out_cnt.data_ptr(),
        sizeof(uint), cudaMemcpyDeviceToHost);

    return cur_ids.slice(0, 0, num_values);
}




void bind_smpl_functions(py::module_ &m) {
    bind_smpl_interpolate(m);
    bind_smpl_local_to_global(m);
    bind_smpl_lbs_skinning(m);

    m.def("smpl_ids_compaction", &smpl_ids_compaction,
        py::arg("instance_ids"), py::arg("instance_mask"));
}
