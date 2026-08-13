#ifndef XSIMGS_NODE_UTILS_CUH
#define XSIMGS_NODE_UTILS_CUH

#include <ATen/cuda/Atomic.cuh>
#include <toast/utils/vector_math.h>


DEVICE void atomicAdd_float3(float3* dst, const float3& src) {
    auto dst_ = reinterpret_cast<float*>(dst);
    gpuAtomicAdd(dst_, src.x);
    gpuAtomicAdd(dst_ + 1, src.y);
    gpuAtomicAdd(dst_ + 2, src.z);
}

DEVICE void atomicAdd_float4(float4* dst, const float4& src) {
    auto dst_ = reinterpret_cast<float*>(dst);
    gpuAtomicAdd(dst_, src.x);
    gpuAtomicAdd(dst_ + 1, src.y);
    gpuAtomicAdd(dst_ + 2, src.z);
    gpuAtomicAdd(dst_ + 3, src.w);
}

#endif //XSIMGS_NODE_UTILS_CUH
