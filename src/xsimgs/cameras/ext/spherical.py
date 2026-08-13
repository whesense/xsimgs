import torch

from xsimgs._C import (
    SphericalCamera as CSphericalCamera,
    isect_spherical_tiles
)

from ..torch import SphericalCameraTorch
from .ext_camera import FusedRSCamera


class SphericalCamera(FusedRSCamera, SphericalCameraTorch):
    ext_cls = CSphericalCamera

    @property
    def _pack_data(self) -> list[torch.Tensor]:
        return super()._pack_data + [
            self.limits.view(4).contiguous(),
        ]

    def isect_gs_tiles(
            self,
            points: torch.Tensor,
            scale: torch.Tensor,
            rotation: torch.Tensor,
            width: int,
            height: int,

            flags: torch.Tensor,
            radii: torch.Tensor,
            means2d: torch.Tensor,
            depths: torch.Tensor,

            points_velocity: torch.Tensor = None,
            opacity: torch.Tensor = None,

            tile_size_x: int = 16,
            tile_size_y: int = 16,

            ut_alpha: float = 0.1,
            ut_beta: float = 2.0,
            ut_kappa: float = 0.0,

            sort: bool = True,
            double_buffer: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return isect_spherical_tiles(
            camera_pack=self.pack,
            points=points,
            points_velocity=points_velocity,
            scale=scale,
            quat=rotation,
            opacity=opacity,
            flag=flags,
            means2d=means2d,
            radii=radii,
            depths=depths,
            tile_size_x=tile_size_x,
            tile_size_y=tile_size_y,
            img_width=width,
            img_height=height,
            ut_alpha=ut_alpha,
            ut_beta=ut_beta,
            ut_kappa=ut_kappa,
            sort=sort,
            double_buffer=double_buffer
        )


