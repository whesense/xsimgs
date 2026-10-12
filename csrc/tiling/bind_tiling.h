#ifndef XSIMGS_BIND_TILING_H
#define XSIMGS_BIND_TILING_H

#include <torch/csrc/utils/pybind.h>

void bind_regular_tiling(py::module_& m);
void bind_lidar_tiling(py::module_& m);
void bind_spherical_tiling(py::module_& m);
void bind_common_tiling(py::module_& m);
void bind_mask_radii(py::module_& m);

void bind_tiling(py::module_& m);

#endif //XSIMGS_BIND_TILING_H
