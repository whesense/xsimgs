#include "sh_bind.h"

#include "sh_fwd.h"
#include "sh_bwd.h"

void bind_sh(py::module_ &m) {
    bind_sh_fwd(m);
    bind_sh_bwd(m);
}