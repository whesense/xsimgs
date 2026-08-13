import numpy as np
import torch

from xsimgs._C import (
    LidarCamera as CLidarCamera,
    isect_lidar_tiles
)


from ..torch import LidarCameraTorch
from .ext_camera import FusedRSCamera


def data_addr_tensor(src_t: torch.Tensor) -> torch.Tensor:
    data_ptr = src_t.data_ptr()
    b1 = data_ptr & 0xffffffff
    b2 = data_ptr >> 32
    arr = np.array([b1, b2], dtype=np.uint32).view(np.float32)
    return torch.tensor(arr, device=src_t.device)


class LidarCamera(FusedRSCamera, LidarCameraTorch):
    ext_cls = CLidarCamera

    @property
    def _pack_data(self) -> list[torch.Tensor]:
        if not self.beam_v.is_contiguous():
            self.beam_v = self.beam_v.contiguous()

        return super()._pack_data + [
            self.limits.view(4).contiguous(),
            data_addr_tensor(self.beam_v)
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
        elev_boundaries = self.beam_v[::tile_size_y].contiguous()

        return isect_lidar_tiles(
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
            elev_boundaries=elev_boundaries,
            tile_size_x=tile_size_x,
            img_width=width,
            img_height=height,
            ut_alpha=ut_alpha,
            ut_beta=ut_beta,
            ut_kappa=ut_kappa,
            sort=sort,
            double_buffer=double_buffer
        )
