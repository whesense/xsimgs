import torch
from toast.transforms import LinearMotion

from ..base import RollingShutter, CameraNameType
from .spherical import SphericalCameraTorch


def cornering_values_from_pixel_values(
        vmin: torch.Tensor | float,
        vmax: torch.Tensor | float,
        img_size: int
):
    rescale = 0.5 / (img_size - 0.5)
    b = (vmin - vmax * rescale) / (1 - rescale)
    a = 2 * (vmin - b)
    vmin_new = b
    vmax_new = a * img_size + b
    return vmin_new, vmax_new


class LidarCameraTorch(SphericalCameraTorch):
    _buffers = SphericalCameraTorch._buffers + ["beam_v"]

    beam_v: torch.Tensor

    @classmethod
    def create(
            cls,
            world_se3_camera: LinearMotion,
            shutter: RollingShutter,
            beam_angles: torch.Tensor,
            phi_min: torch.Tensor,
            phi_range: torch.Tensor,
            camera_name: CameraNameType = None
    ):
        num_beams = beam_angles.shape[-1]
        theta_min, theta_max = cornering_values_from_pixel_values(
            vmin=beam_angles.amax(dim=-1),
            vmax=beam_angles.amin(dim=-1),
            img_size=num_beams
        )
        theta_min = theta_min.clamp(-torch.pi/2, torch.pi/2)
        theta_max = theta_max.clamp(-torch.pi/2, torch.pi/2)
        theta_range = theta_max - theta_min

        limits = torch.stack([
            phi_min, phi_range, theta_min, theta_range
        ], dim=-1)
        beam_v = (beam_angles.contiguous() - theta_min) / theta_range

        return cls(
            world_se3_camera=world_se3_camera,
            shutter=shutter,
            limits=limits,
            beam_v=beam_v,
            camera_name=camera_name
        )

    def __init__(
            self,
            world_se3_camera: LinearMotion,
            shutter: RollingShutter,
            limits: torch.Tensor,
            beam_v: torch.Tensor,
            camera_name: CameraNameType = None
    ):
        super().__init__(
            world_se3_camera=world_se3_camera,
            shutter=shutter,
            limits=limits,
            camera_name=camera_name
        )
        self.beam_v = beam_v

    @property
    def num_beams(self):
        return self.beam_v.shape[-1]

    def _check_height(self, height: int):
        assert height == self.num_beams, \
            ("Image height must match with the number "
             "of LiDAR beams (num_beams={}, height={})").format(
                self.num_beams, height)

    def pixel_to_uv(self, xy: torch.Tensor, width: int, height: int):
        self._check_height(height)

        u = (xy[..., 0] + 0.5) / width
        v = self.beam_v.view(-1, 1).expand_as(u)
        return torch.cat([u, v], dim=-1)

