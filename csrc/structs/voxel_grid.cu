#include <torch/csrc/utils/pybind.h>
#include "voxel_grid.cuh"
#include <toast/math/se3.h>
#include <structs/VoxelGrid3D.h>
#include <toast/utils/launch_helpers.h>
#include <toast/utils/tensor_helpers.h>





DEFINE_KERNEL_CAPTURE(launch_vg_voxel_coord_flt, {
    const auto [out, p] = ptrs;
    const auto& [vmin, vsize] = captures;
    out[offsets[0]] = (p[offsets[1]] - vmin) / vsize;
})


at::Tensor vg_voxel_coord_flt(
    const at::Tensor& vmin,
    const at::Tensor& vsize,
    const at::Tensor& points
) {
    TORCH_CHECK(points.size(-1) == 3, "points must be [..., 3]");
    all_contiguous(points);

    auto out = at::empty_like(points);
    auto iter = make_tensor_iterator({out}, {points});
    check_operand_last_dims(iter, {3, 3});

    DISPATCH_KERNEL_CAPTURE(launch_vg_voxel_coord_flt,
        std::make_tuple(tensor_to_vec3<dtype>(vmin), tensor_to_vec3<dtype>(vsize)),
        vec3*, const vec3*)
    return out;
}


DEFINE_KERNEL_CAPTURE(launch_vg_voxel_coord_clamped, {
    const auto [out, p] = ptrs;
    const auto& [grid] = captures;
    const uint3 c = grid.voxel_coord_clamped(p[offsets[1]]);
    out[offsets[0]] = make_int3(c.x, c.y, c.z);
})


at::Tensor vg_voxel_coord_clamped(
    const at::Tensor& vmin,
    const at::Tensor& vmax,
    const at::Tensor& vsize,
    const at::Tensor& points
) {
    TORCH_CHECK(points.size(-1) == 3, "points must be [..., 3]");
    check_same_dtype(vmin, vmax, vsize, points);

    auto out = at::empty(points.sizes(), points.options().dtype(at::kInt));
    auto iter = make_tensor_iterator(
        {out}, {points}, {-1}, /*check_same_dtype=*/false);
    check_operand_last_dims(iter, {3, 3});
    DISPATCH_KERNEL_CAPTURE(
        launch_vg_voxel_coord_clamped,
        std::make_tuple(VoxelGrid3D<dtype>(
            tensor_to_vec3<dtype>(vmin),
            tensor_to_vec3<dtype>(vmax),
            tensor_to_vec3<dtype>(vsize)
        )),
        int3*, const vec3*)
    return out;
}


DEFINE_KERNEL_CAPTURE(launch_vg_voxel_coord_int, {
    const auto [out, p] = ptrs;
    const auto& [grid] = captures;
    const uint3 c = grid.voxel_coord(p[offsets[1]]);
    out[offsets[0]] = make_int3(c.x, c.y, c.z);
})

at::Tensor vg_voxel_coord_int(
    const at::Tensor& vmin,
    const at::Tensor& vmax,
    const at::Tensor& vsize,
    const at::Tensor& points
) {
    TORCH_CHECK(points.size(-1) == 3, "points must be [..., 3]");
    check_same_dtype(vmin, vmax, vsize, points);

    auto out = at::empty(points.sizes(), points.options().dtype(at::kInt));
    auto iter = make_tensor_iterator({out}, {points}, {-1}, /*check_same_dtype=*/false);
    check_operand_last_dims(iter, {3, 3});
    DISPATCH_KERNEL_CAPTURE(
        launch_vg_voxel_coord_int,
        std::make_tuple(VoxelGrid3D<dtype>(
            tensor_to_vec3<dtype>(vmin),
            tensor_to_vec3<dtype>(vmax),
            tensor_to_vec3<dtype>(vsize)
        )), int3*, const vec3*)
    return out;
}


DEFINE_KERNEL_CAPTURE(launch_vg_voxel_offset, {
    const auto [out, p] = ptrs;
    const auto& [grid] = captures;
    out[offsets[0]] = grid._voxel_offset(grid.voxel_coord_clamped(p[offsets[1]]));
})


at::Tensor vg_voxel_offset(
    const at::Tensor& vmin,
    const at::Tensor& vmax,
    const at::Tensor& vsize,
    const at::Tensor& points
) {
    check_same_dtype(vmin, vmax, vsize, points);
    auto shape = points.sizes().vec();
    shape.pop_back(); shape.push_back(1);
    auto out = at::empty(shape, points.options().dtype(at::kLong));
    auto iter = make_tensor_iterator({out}, {points}, {-1}, /*check_same_dtype=*/false);
    check_operand_last_dims(iter, {1, 3});
    DISPATCH_KERNEL_CAPTURE(
        launch_vg_voxel_offset,
        std::make_tuple(VoxelGrid3D<dtype>(
            tensor_to_vec3<dtype>(vmin),
            tensor_to_vec3<dtype>(vmax),
            tensor_to_vec3<dtype>(vsize)
        )),
        int64_t*, const vec3*)
    return out.squeeze(-1);
}


DEFINE_KERNEL_CAPTURE(launch_vg_is_inside, {
    const auto [out, p] = ptrs;
    const auto& [vmin, vmax] = captures;
    const auto& pt = p[offsets[1]];
    out[offsets[0]] = (pt.x >= vmin.x) & (pt.x < vmax.x)
                    & (pt.y >= vmin.y) & (pt.y < vmax.y)
                    & (pt.z >= vmin.z) & (pt.z < vmax.z);
})

at::Tensor vg_is_inside(
    const at::Tensor& vmin,
    const at::Tensor& vmax,
    const at::Tensor& points
) {
    check_same_dtype(vmin, vmax, points);
    auto shape = points.sizes().vec();
    shape.pop_back(); shape.push_back(1);

    auto out = at::empty(shape, points.options().dtype(at::kBool));
    auto iter = make_tensor_iterator(
        {out}, {points}, {-1}, /*check_same_dtype=*/false);
    check_operand_last_dims(iter, {1, 3});
    DISPATCH_KERNEL_CAPTURE(launch_vg_is_inside,
        std::make_tuple(
            tensor_to_vec3<dtype>(vmin),
            tensor_to_vec3<dtype>(vmax)),
        bool*, const vec3*)
    return out.squeeze(-1);
}


void bind_voxel_grid(pybind11::module& m) {
    m.def("vg_voxel_coord_flt", &vg_voxel_coord_flt,
        py::arg("vmin"), py::arg("vsize"), py::arg("points"));
    m.def("vg_voxel_coord_clamped", &vg_voxel_coord_clamped,
        py::arg("vmin"), py::arg("vmax"), py::arg("vsize"), py::arg("points"));
    m.def("vg_voxel_coord_int", &vg_voxel_coord_int,
        py::arg("vmin"), py::arg("vmax"), py::arg("vsize"), py::arg("points"));
    m.def("vg_voxel_offset", &vg_voxel_offset,
        py::arg("vmin"), py::arg("vmax"), py::arg("vsize"), py::arg("points"));
    m.def("vg_is_inside", &vg_is_inside,
        py::arg("vmin"), py::arg("vmax"), py::arg("points"));
}
