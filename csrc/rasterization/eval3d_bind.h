#ifndef XSIMGS_EVAL3D_BIND_H
#define XSIMGS_EVAL3D_BIND_H

#include <torch/csrc/utils/pybind.h>

void bind_rasterization_eval3d_fwd(py::module& m);
void bind_rasterization_eval3d_bwd(py::module& m);
void bind_rasterization_eval3d_dynamic_fwd(py::module& m);
void bind_rasterization_eval3d_dynamic_bwd(py::module& m);

void bind_rasterization_eval3d(py::module& m);

#endif //XSIMGS_EVAL3D_BIND_H
