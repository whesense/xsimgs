#ifndef VOXELGRID3D_H
#define VOXELGRID3D_H

#include <stdexcept>
#include "ROIBox3D.h"


inline double3 operator/(const double3& a, const double3& b) {
    return {a.x/b.x, a.y/b.y, a.z/b.z};
}


template <typename dtype>
struct VoxelGrid3D: ROIBox3D<dtype> {
    using vec3 = typename dtype::vec3;

    vec3 voxel_size;
    uint3 voxel_dims;

    HOSTDEVICE VoxelGrid3D(
        const vec3& vmin,
        const vec3& vmax,
        const vec3& voxel_size,
        const uint3& voxel_dims
    ): ROIBox3D<dtype>(vmin, vmax), voxel_size(voxel_size), voxel_dims(voxel_dims) {}

    VoxelGrid3D(const vec3& vmin, const vec3& vmax, const vec3& voxel_size):
        ROIBox3D<dtype>(vmin, vmax), voxel_size(voxel_size)
    {
        voxel_dims = dims();
        uint64_t total_voxels;
        bool overflow = __builtin_umull_overflow(voxel_dims.x, voxel_dims.y, &total_voxels);
        overflow |= __builtin_umull_overflow(total_voxels, voxel_dims.z, &total_voxels);
        if (overflow || ((total_voxels >> 63) != 0)) {
            throw std::runtime_error(
                "VoxelGrid3D: 64-bit index overflow "
                "(voxel_dims=["+std::to_string(voxel_dims.x)+", "+
                std::to_string(voxel_dims.y)+", "+
                std::to_string(voxel_dims.z)+"])");
        }
    }

    // VoxelGrid3D(const tuple_float3& vmin, const tuple_float3& vmax, const tuple_float3& _voxel_size):
    //     VoxelGrid3D(tuple_to_float3(vmin), tuple_to_float3(vmax), tuple_to_float3(_voxel_size)) {}

    [[nodiscard]] HOSTDEVICE uint3 dims() const {
        return voxel_coord(ROIBox3D<dtype>::vmax) + 1;
    }

    [[nodiscard]] HOSTDEVICE vec3 voxel_coord_flt(const vec3& point) const {
        return (point - ROIBox3D<dtype>::vmin) / voxel_size;
    }

    [[nodiscard]] HOSTDEVICE int64_t _voxel_offset(const uint3& voxel) const {
        const auto z_dim = static_cast<int64_t>(voxel_dims.z);
        const auto y_dim = static_cast<int64_t>(voxel_dims.y);
        return
            static_cast<int64_t>(voxel.z) +
                static_cast<int64_t>(voxel.y) * z_dim +
                    static_cast<int64_t>(voxel.x) * z_dim * y_dim;
    }

    [[nodiscard]] HOSTDEVICE uint3 voxel_coord(const vec3& point) const {
        auto c = voxel_coord_flt(point);
        return make_uint3(
            static_cast<uint>(dtype::floor(c.x)),
            static_cast<uint>(dtype::floor(c.y)),
            static_cast<uint>(dtype::floor(c.z)));
    }

    [[nodiscard]] HOSTDEVICE uint3 voxel_coord_clamped(const vec3& point) const {
        return clamp(voxel_coord(point), make_uint3(0), voxel_dims - 1u);
    }

    template <typename T>
    [[nodiscard]] HOSTDEVICE T* voxel_ptr(const uint3 &voxel_idx, T *data_ptr) const {
        return &data_ptr[_voxel_offset(voxel_idx)];
    }

    template <typename T>
    [[nodiscard]] HOSTDEVICE const T* voxel_ptr(const uint3 &voxel_idx, const T *data_ptr) const {
        return voxel_ptr(voxel_idx, data_ptr);
    }

    template <typename T>
    [[nodiscard]] HOSTDEVICE T voxel_data(const uint3& voxel_idx, const T* data_ptr) const {
        return data_ptr[_voxel_offset(voxel_idx)];
    }
};


#endif //VOXELGRID3D_H
