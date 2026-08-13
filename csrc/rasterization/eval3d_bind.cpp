#include "eval3d_bind.h"

#include "common.h"

void bind_rasterization_eval3d(py::module& m) {
    bind_rasterization_eval3d_fwd(m);
    bind_rasterization_eval3d_bwd(m);

    bind_rasterization_eval3d_dynamic_fwd(m);
    bind_rasterization_eval3d_dynamic_bwd(m);

#define iter_macro(v) v,
    m.def("eval3d_supported_ndims", []() -> std::vector<int> {
        return {ndim_cases(iter_macro)};
    });
#undef iter_macro

}