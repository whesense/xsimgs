#ifndef XSIMGS_MCMC_NOISE_H
#define XSIMGS_MCMC_NOISE_H

#include <torch/extension.h>

void bind_mcmc_noise(py::module_& m);

#endif //XSIMGS_MCMC_NOISE_H
