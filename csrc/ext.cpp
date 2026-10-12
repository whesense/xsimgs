#include <torch/csrc/utils/pybind.h>

#include "cameras/cameras_bind.h"
#include "color/sh_bind.h"
#include "nodes/bind_nodes.h"
#include "rasterization/eval3d_bind.h"
#include "strategy/mcmc_noise.h"
#include "structs/bind_structs.h"
#include "tiling/bind_tiling.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    bind_cameras(m);
    bind_tiling(m);
    bind_rasterization_eval3d(m);
    bind_sh(m);
    bind_nodes(m);
    bind_structs(m);
    bind_mcmc_noise(m);
}