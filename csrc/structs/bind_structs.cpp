

#include "bind_structs.h"

#include "voxel_grid.cuh"


void bind_structs(py::module_ &m) {
    bind_oriented_box(m);
    bind_voxel_grid(m);
}
