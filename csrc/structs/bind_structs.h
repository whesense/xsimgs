

#ifndef XSIMGS_BIND_STRUCTS_H
#define XSIMGS_BIND_STRUCTS_H

#include <torch/extension.h>

void bind_structs(py::module_& m);

void bind_oriented_box(py::module_& m);

#endif //XSIMGS_BIND_STRUCTS_H
