import torch
from toast import quat_apply, quat_apply_inv, quat_std

from xsimgs._C import mcmc_add_noise as _mcmc_add_noise


def add_mcmc_noise(
        positions: torch.Tensor,
        density: torch.Tensor,
        rotation: torch.Tensor,
        scale: torch.Tensor,
        k: float,
        x0: float,
        noise_scale: float,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
) -> torch.Tensor:
    if noise is None:
        noise = torch.randn(
            positions.shape, dtype=positions.dtype,
            device=positions.device, generator=generator
        )
    _mcmc_add_noise(
        positions, density, rotation, scale, noise,
        float(k), float(x0), float(noise_scale)
    )
    return positions


def add_mcmc_noise_torch(
        positions: torch.Tensor,
        density: torch.Tensor,
        rotation: torch.Tensor,
        scale: torch.Tensor,
        k: float,
        x0: float,
        noise_scale: float,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
) -> torch.Tensor:
    if noise is None:
        noise = torch.randn(
            positions.shape, dtype=positions.dtype,
            device=positions.device, generator=generator
        )

    densities = 1.0 - torch.sigmoid(density)
    d_sig = torch.sigmoid(k * (densities - x0))

    q = quat_std(rotation)
    scale_sq = torch.exp(scale) ** 2

    n = noise * d_sig * noise_scale
    delta = quat_apply(q, scale_sq * quat_apply_inv(q, n))
    return positions + delta
