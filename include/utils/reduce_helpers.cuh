#include <toast/utils/vector_math.h>
#include <cub/cub.cuh>


// TODO: REMOVE DEPENDENCY ON THIS CUDA_NUM_THREADS!!! THIS WILL PRODUCE ERROR IF NUMBER OF THREADS MISMATCH LAUNCH CONFIG
#define CUDA_NUM_THREADS 1024

#define block_reduce_(out_ptr, type, field_offset, value, ...)\
    {\
        type tmp = cub::BlockReduce<type, CUDA_NUM_THREADS>(sh_mem.type##_storage).Sum(value);\
        if (threadIdx.x == 0) {\
            float* ptr = reinterpret_cast<float*>(out_ptr + (field_offset) * gridDim.x) + blockIdx.x;\
            __VA_ARGS__\
        }\
        __syncthreads();\
    }

#define block_reduce_float4(out_ptr, field_offset, value)\
    block_reduce_(out_ptr, float4, field_offset, value, ptr[0] = tmp.x; ptr[gridDim.x] = tmp.y; ptr[2*gridDim.x] = tmp.z; ptr[3*gridDim.x] = tmp.w;)

#define block_reduce_float3(out_ptr, field_offset, value)\
    block_reduce_(out_ptr, float3, field_offset, value, ptr[0] = tmp.x; ptr[gridDim.x] = tmp.y; ptr[2*gridDim.x] = tmp.z;)

#define block_reduce_float2(out_ptr, field_offset, value)\
    block_reduce_(out_ptr, float2, field_offset, value, ptr[0] = tmp.x; ptr[gridDim.x] = tmp.y;)

#define block_reduce_float(out_ptr, field_offset, value)\
    block_reduce_(out_ptr, float, field_offset, value, ptr[0] = tmp;)


template <uint num_threads>
union BlockReduceType{
    typename cub::BlockReduce<float4, num_threads>::TempStorage float4_storage;
    typename cub::BlockReduce<float3, num_threads>::TempStorage float3_storage;
    typename cub::BlockReduce<float2, num_threads>::TempStorage float2_storage;
    typename cub::BlockReduce<float, num_threads>::TempStorage float_storage;
};
