import torch

from xsimgs._C import mask_radii, isect_offset_encode, eval3d_supported_ndims

from .rasterize import ut_gs_rasterize
from .rasterize_dynamic import ut_gs_dynamic_rasterize
from ..cameras.ext.ext_camera import FusedRSCamera


def ut_gs_tiling(
        camera: FusedRSCamera,
        points: torch.Tensor,
        scale: torch.Tensor,
        rotation: torch.Tensor,
        width: int,
        height: int,
        points_velocity: torch.Tensor = None,
        opacity: torch.Tensor = None,
        packed_particle_mask: torch.Tensor = None,
        tile_size_x: int = 16,
        tile_size_y: int = 16,
        ut_alpha: float = 0.1,
        ut_beta: float = 2.0,
        ut_kappa: float = 0.0,
        use_phm: bool = None
) -> dict[str, torch.Tensor]:
    ctx = camera._freeze_ctx().__enter__()

    flags, radii, means2d, depth = camera.ut_gs_projection(
        points=points,
        points_velocity=points_velocity,
        scale=scale,
        rotation=rotation,
        opacity=opacity,
        width=width,
        height=height,
        ut_alpha=ut_alpha,
        ut_beta=ut_beta,
        ut_kappa=ut_kappa,
        use_phm=use_phm
    )
    if packed_particle_mask is not None:
        mask_radii(packed_particle_mask, radii)

    tiles_per_gs, isect_ids, flatten_ids = camera.isect_gs_tiles(
        points=points,
        points_velocity=points_velocity,
        scale=scale,
        rotation=rotation,
        opacity=opacity,

        flags=flags,
        means2d=means2d,
        radii=radii,
        depths=depth,

        width=width,
        height=height,
        ut_alpha=ut_alpha,
        ut_beta=ut_beta,
        ut_kappa=ut_kappa,
        tile_size_x=tile_size_x,
        tile_size_y=tile_size_y,
        sort=True,
        double_buffer=True
    )

    n_tiles_width = (width + tile_size_x - 1) // tile_size_x
    n_tiles_height = (height + tile_size_y - 1) // tile_size_y

    tile_gs_offsets = isect_offset_encode(
        isect_ids, tile_width=n_tiles_width, tile_height=n_tiles_height
    )

    return dict(
        means2d=means2d,
        radii=radii,
        depth=depth,
        isect_offsets=tile_gs_offsets,
        flatten_ids=flatten_ids
    )



def ut_gs_render(
        camera: FusedRSCamera,
        points: torch.Tensor,
        scale: torch.Tensor,
        rotation: torch.Tensor,
        opacity: torch.Tensor,
        width: int,
        height: int,
        colors: torch.Tensor = None,
        points_velocity: torch.Tensor = None,
        packed_particle_mask: torch.Tensor = None,
        tile_size_x: int = 16,
        tile_size_y: int = 16,
        ut_alpha: float = 0.1,
        ut_beta: float = 2.0,
        ut_kappa: float = 0.0,
        use_phm: bool = None,
        ray_o: torch.Tensor = None,
        ray_d: torch.Tensor = None,
        ray_time_delta: torch.Tensor = None
):
    assert camera.device == points.device,\
        ("Camera must be on the same device as gaussian tensors. "
         "Given camera on {}, points on {}".format(
            camera.device, points.device))
    if colors is not None:
        assert colors.shape[-1] in eval3d_supported_ndims(), \
            ("Color feature dimension = {} is not supported. Supported CDIMs=[{}]."
             "To add support for your particular case, you can add line "
             "'macro(YOUR_CDIM)\\' into csrc/rasterization/common.h "
             "and recompile xsimgs".format(
                colors.shape[-1],
                ', '.join([str(v) for v in eval3d_supported_ndims()])))

    ctx = camera._freeze_ctx().__enter__()

    points = points.contiguous()
    scale = scale.contiguous()
    rotation = rotation.contiguous()
    opacity = opacity.contiguous()
    colors = colors.contiguous() if colors is not None else None
    points_velocity = (
        points_velocity.contiguous() if points_velocity is not None else None
    )

    tile_data = ut_gs_tiling(
        camera=camera,
        points=points,
        scale=scale,
        rotation=rotation,
        width=width,
        height=height,
        points_velocity=points_velocity,
        opacity=opacity,
        packed_particle_mask=packed_particle_mask,
        tile_size_x=tile_size_x,
        tile_size_y=tile_size_y,
        ut_alpha=ut_alpha,
        ut_beta=ut_beta,
        ut_kappa=ut_kappa,
        use_phm=use_phm
    )

    is_scene_static = points_velocity is None
    if ray_o is None or ray_d is None:
        ray_o, ray_d = camera.camera_rays(
            width=width,
            height=height,
            normalized_rays=True,
            return_origins=True,
            world_space=True
        )
    kwargs = dict(
        positions=points,
        rotation=rotation,
        scale=scale,
        density=opacity,
        colors=colors,
        ray_o=ray_o,
        ray_d=ray_d,
        isect_offsets=tile_data['isect_offsets'],
        flatten_ids=tile_data['flatten_ids'],
        tile_size_x=tile_size_x,
        tile_size_y=tile_size_y
    )

    if is_scene_static:
        color, depth, alpha = ut_gs_rasterize(**kwargs)
    else:
        if ray_time_delta is None:
            ray_time_delta = camera.image_time_delta(ray_o)
        color, depth, alpha = ut_gs_dynamic_rasterize(
            **kwargs, velocity=points_velocity,
            ray_time_delta=ray_time_delta
        )

    meta = dict(
        radii=tile_data['radii'],
        width=width,
        height=height,
    )

    return color, depth, alpha, meta
