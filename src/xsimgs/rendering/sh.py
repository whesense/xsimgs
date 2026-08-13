import torch

from xsimgs._C import (
    spherical_harmonics_fused_fwd,
    spherical_harmonics_fused_bwd
)

from toast.ops.ext.ops_utils import apply_grad_mask


class SphericalHarmonicsFused(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            coeffs: torch.Tensor,
            positions: torch.Tensor,
            origin: torch.Tensor,
            degree_to_use: int,
            mask: torch.Tensor | None = None,
            layout_3kn: bool = False,
            coeffs_dc: torch.Tensor | None = None,
    ):
        ctx.layout_3kn = layout_3kn
        ctx.degree_to_use = degree_to_use
        ctx.save_for_backward(origin, positions, coeffs, mask, coeffs_dc)

        return spherical_harmonics_fused_fwd(
            degree_to_use, origin, positions, coeffs, mask, layout_3kn, coeffs_dc
        )

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        origin, positions, coeffs, mask, coeffs_dc = ctx.saved_tensors

        d_coeffs, d_coeffs_dc, d_positions, d_origin = spherical_harmonics_fused_bwd(
            ctx.degree_to_use,
            origin, positions, coeffs, mask, ctx.layout_3kn, grad_out, coeffs_dc
        )
        # Order matches the forward inputs:
        # coeffs, positions, origin, degree_to_use, mask, layout_3kn, coeffs_dc
        grads = (d_coeffs, d_positions, d_origin, None, None, None, d_coeffs_dc)
        return apply_grad_mask(ctx, grads)


def spherical_harmonics_fused(
        coeffs: torch.Tensor,
        positions: torch.Tensor,
        origin: torch.Tensor,
        degree_to_use: int,
        mask: torch.Tensor | None = None,
        layout_3kn: bool = False,
        coeffs_dc: torch.Tensor | None = None,
):
    return SphericalHarmonicsFused.apply(
        coeffs, positions, origin, degree_to_use, mask, layout_3kn, coeffs_dc)


def sh_degree_to_bands(degree: int):
    return (degree + 1) ** 2

def sh_degree_to_specular_dim(degree: int):
    """Number of dimensions used by SH of deg [1...degree], inclusive"""
    return 3 * (sh_degree_to_bands(degree) - 1)

def rgb_to_sh0(rgb: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def sh0_to_rgb(sh: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return sh * C0 + 0.5
