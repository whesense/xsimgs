#include "cameras_bind.h"
#include "spherical.h"

#include <cameras/bind.h>


void bind_spherical(py::module_& m) {
    pybind_camera<RegularSphericalCamera>(m, "SphericalCamera");
    pybind_camera<LidarCamera>(m, "LidarCamera");
}