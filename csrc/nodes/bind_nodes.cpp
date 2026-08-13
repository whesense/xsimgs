#include "bind_nodes.h"

#include "container.h"
#include "gs_masking.h"
#include "img_layout.h"
#include "smpl.h"


void bind_nodes(py::module_ &m) {
    bind_image_functions(m);
    bind_gs_masking_functions(m);
    bind_container_functions(m);
    bind_smpl_functions(m);
}

