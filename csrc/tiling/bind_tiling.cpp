#include "bind_tiling.h"

void bind_tiling(py::module_& m) {
    bind_common_tiling(m);
    bind_regular_tiling(m);
    bind_lidar_tiling(m);
    bind_spherical_tiling(m);
    bind_mask_radii(m);
}

