#ifndef XSIMGS_TILE_SHARED_MEMORY_H
#define XSIMGS_TILE_SHARED_MEMORY_H

#include <cuda_runtime.h>
#include <c10/util/Exception.h>


template <typename SharedMemoryType>
void set_max_shared_mem(
    const void* fn,
    const uint32_t tile_width,
    const uint32_t tile_height
) {
    uint32_t shmem_size = SharedMemoryType::mem_size(tile_width, tile_height);
    if (cudaFuncSetAttribute(
        fn,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem_size
    ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }
}

#define SOA_DECLARE_ARRAY(type, name, size_mult) type* name;
#define SOA_INIT_PTR(type, name, size_mult) \
    name = (type*)(ptr + offset); offset += block_size * sizeof(type) * size_mult;
#define SOA_ADD_SIZE(type, name, size_mult) total += sizeof(type) * size_mult;

#define SHARED_MEM_LAYOUT(layout_name, STRUCT_FIELDS) \
struct layout_name { \
    \
    STRUCT_FIELDS(SOA_DECLARE_ARRAY) \
    \
    __device__ layout_name(const char* ptr, uint32_t block_size) { \
        uint32_t offset = 0; \
        STRUCT_FIELDS(SOA_INIT_PTR) \
    }\
    \
    static constexpr size_t elementSize() { \
        size_t total = 0; \
        STRUCT_FIELDS(SOA_ADD_SIZE) \
        return total; \
    } \
    \
    static uint32_t mem_size(const uint32_t tile_width, const uint32_t tile_height) { \
        return tile_width * tile_height * elementSize(); \
    } \
};


#endif //XSIMGS_TILE_SHARED_MEMORY_H
