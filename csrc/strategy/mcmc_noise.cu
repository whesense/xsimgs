// MCMC position-noise injection, as described in Kheradmand et al.,
// "3D Gaussian Splatting as Markov Chain Monte Carlo" (Eq. 9).

#include "mcmc_noise.h"

#include <toast/utils/vector_math.h>
#include <toast/utils/launch_helpers.h>
#include <toast/utils/tensor_helpers.h>
#include <toast/math/se3.h>


__device__ __forceinline__ float sigmoidf(const float x) {
    return 1.0f / (1.0f + expf(-x));
}


__global__ void mcmc_add_noise_kernel(
    const uint N,
    float3* __restrict__ positions,      // [N, 3] non-activated, updated in place
    const float* __restrict__ density,   // [N, 1] non-activated
    const quatf* __restrict__ rotation,  // [N, 4] non-activated quaternion (wxyz)
    const float3* __restrict__ scale,    // [N, 3] non-activated
    const float3* __restrict__ noise,    // [N, 3] standard-normal samples
    const float k,
    const float x0,
    const float noise_scale              // = noise_lr * positions_lr
) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // densities = 1 - sigmoid(density); d_sig = sigmoid(k * (densities - x0))
    const float densities = 1.0f - sigmoidf(density[idx]);
    const float d_sig = sigmoidf(k * (densities - x0));

    // rotation activation: quaternion .std()
    const quatf q = rotation[idx].std();

    // scale activation: exp, then squared (covariance scaling)
    const float3 s = scale[idx];
    const float3 scale_act = make_float3(expf(s.x), expf(s.y), expf(s.z));
    const float3 scale_sq = scale_act * scale_act;

    // noise = randn * d_sig * noise_lr * lr, then rotate into the gaussian frame,
    // weight by the covariance scale and rotate back:
    //   noise = q.apply(scale_sq * q.apply_inv(noise))
    float3 n = noise[idx] * (d_sig * noise_scale);
    n = q.apply(scale_sq * q.apply_inv(n));

    positions[idx] += n;
}


void mcmc_add_noise(
    torch::Tensor& positions,
    const torch::Tensor& density,
    const torch::Tensor& rotation,
    const torch::Tensor& scale,
    const torch::Tensor& noise,
    const double k,
    const double x0,
    const double noise_scale
) {
    check_same_device(positions, density, rotation, scale, noise);
    all_contiguous(positions, density, rotation, scale, noise);

    const uint N = positions.size(0);
    TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3,
        "positions must be [N, 3]");
    TORCH_CHECK(density.size(0) == N && density.numel() == N,
        "density must be [N, 1] (or [N])");
    TORCH_CHECK(rotation.dim() == 2 && rotation.size(0) == N && rotation.size(1) == 4,
        "rotation must be [N, 4]");
    TORCH_CHECK(scale.dim() == 2 && scale.size(0) == N && scale.size(1) == 3,
        "scale must be [N, 3]");
    TORCH_CHECK(noise.sizes() == positions.sizes(),
        "noise must have the same shape as positions [N, 3]");

    if (N == 0) return;

    launch_kernel_cuda(N, positions.device(),
        mcmc_add_noise_kernel,
        N,
        (float3*)positions.data_ptr(),
        (const float*)density.data_ptr(),
        (const quatf*)rotation.data_ptr(),
        (const float3*)scale.data_ptr(),
        (const float3*)noise.data_ptr(),
        (float)k, (float)x0, (float)noise_scale
    );
}


void bind_mcmc_noise(py::module_& m) {
    m.def("mcmc_add_noise", &mcmc_add_noise,
        py::arg("positions"),
        py::arg("density"),
        py::arg("rotation"),
        py::arg("scale"),
        py::arg("noise"),
        py::arg("k"),
        py::arg("x0"),
        py::arg("noise_scale")
    );
}
