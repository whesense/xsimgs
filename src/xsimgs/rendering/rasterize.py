import torch
from toast.ops.ext import apply_grad_mask

from xsimgs._C import rasterize_eval3d_fwd, rasterize_eval3d_bwd

class UTGSRasterize(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            positions: torch.Tensor,
            rotation: torch.Tensor,
            scale: torch.Tensor,
            density: torch.Tensor,
            colors: torch.Tensor,
            ray_o: torch.Tensor,
            ray_d: torch.Tensor,
            isect_offsets: torch.Tensor,
            flatten_ids: torch.Tensor,
            tile_size_x: int,
            tile_size_y: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        height, width = ray_o.shape[:2]
        out_rgb, out_depth, out_alpha, last_ids = rasterize_eval3d_fwd(
            means=positions,
            quats=rotation,
            scales=scale,
            opacities=density.view(1, -1),
            colors=colors.unsqueeze(0) if colors is not None else None,
            ray_origs=ray_o,
            ray_dirs=ray_d,
            image_width=width,
            image_height=height,
            tile_size_width=tile_size_x,
            tile_size_height=tile_size_y,
            tile_offsets=isect_offsets,
            flatten_ids=flatten_ids
        )
        ctx.tile_size_x = tile_size_x
        ctx.tile_size_y = tile_size_y
        ctx.save_for_backward(
            positions, rotation, scale, density, colors,
            ray_o, ray_d,
            isect_offsets, flatten_ids,
            out_alpha, last_ids
        )
        return out_rgb, out_depth, out_alpha

    @staticmethod
    def backward(
            ctx,
            grad_rgb: torch.Tensor,
            grad_depth: torch.Tensor,
            grad_alpha: torch.Tensor
    ):
        (
            positions, rotation, scale, density, colors,
            ray_o, ray_d,
            isect_offsets, flatten_ids,
            out_alpha, last_ids
        ) = ctx.saved_tensors
        tile_size_x = ctx.tile_size_x
        tile_size_y = ctx.tile_size_y
        # v_points, v_quat, v_scale, v_opacity, v_colors, v_ray_o, v_ray_d
        grads = rasterize_eval3d_bwd(
            positions, scale, rotation, density, colors, ray_o, ray_d,
            tile_size_x, tile_size_y, isect_offsets, flatten_ids,
            out_alpha, last_ids,
            grad_rgb.contiguous() if grad_rgb is not None else None,
            grad_depth.contiguous(),
            grad_alpha.contiguous()
        )
        return apply_grad_mask(ctx, list(grads) + [None, None, None, None])


def ut_gs_rasterize(
        positions: torch.Tensor,
        rotation: torch.Tensor,
        scale: torch.Tensor,
        density: torch.Tensor,
        colors: torch.Tensor | None,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        isect_offsets: torch.Tensor,
        flatten_ids: torch.Tensor,
        tile_size_x: int,
        tile_size_y: int
):
    return UTGSRasterize.apply(
        positions,
        rotation,
        scale,
        density,
        colors,
        ray_o,
        ray_d,
        isect_offsets,
        flatten_ids,
        tile_size_x,
        tile_size_y
    )
