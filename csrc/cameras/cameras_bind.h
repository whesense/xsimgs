#ifndef CAMERAS_BIND_H
#define CAMERAS_BIND_H

#include <torch/extension.h>

void bind_perspective(py::module_& m);
void bind_distorted_perspective(py::module_& m);
void bind_fisheye(py::module_& m);
void bind_spherical(py::module_& m);

inline void bind_cameras(pybind11::module& m) {
    bind_perspective(m);
    bind_distorted_perspective(m);
    bind_fisheye(m);
    bind_spherical(m);
}

#endif //XSIMGS_CAMERAS_H
