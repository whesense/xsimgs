#ifndef XSIMGS_SMPL_H
#define XSIMGS_SMPL_H

#include <torch/extension.h>

#define NUM_JOINTS 24


void bind_smpl_functions(py::module_& m);

void bind_smpl_local_to_global(py::module_& m);
void bind_smpl_interpolate(py::module_& m);
void bind_smpl_lbs_skinning(py::module_& m);

#endif //XSIMGS_SMPL_H
