import torch
from toast import LinearMotion

from ..base import RSCamera, RollingShutter, CameraNameType


def spherical_to_cartesian(polar: torch.Tensor):
    azimuth, elevation, distance = polar.unbind(dim=-1)
    xy_dist = distance * elevation.cos()
    x = xy_dist * azimuth.sin()
    y = xy_dist * azimuth.cos()
    z = distance * elevation.sin()
    return torch.stack([x, y, z], dim=-1)


def cartesian_to_spherical(p: torch.Tensor):
    r = p.norm(dim=-1)
    x, y, z = p.unbind(dim=-1)

    elevation = (z / r).asin()
    azimuth = torch.atan2(-x, -y) + torch.pi
    return torch.stack([azimuth, elevation, r], dim=-1)



class SphericalCameraTorch(RSCamera):
    _buffers = RSCamera._buffers + ["limits"]

    # 4 angles in radians tensor: [phi_min, phi_range, theta_min, theta_range]
    # phi_min = azimuth angle that corresponds to left corner of image
    # phi_range = phi_max - phi_min = azimuth coverage of image
    #   (in case of full spherical = 2*PI)
    # theta_min, theta_range = elevation angles range
    # NOTE: this range corresponds to the image corners,
    #   not the cornering pixel centers
    limits: torch.Tensor

    def __init__(
            self,
            world_se3_camera: LinearMotion,
            shutter: RollingShutter,
            limits: torch.Tensor = None,
            camera_name: CameraNameType = None
    ):
        super().__init__(
            world_se3_camera=world_se3_camera,
            shutter=shutter,
            camera_name=camera_name
        )
        if limits is None:
            limits = torch.tensor([0.0, 2*torch.pi, -torch.pi/2.0, torch.pi])
            limits = limits.to(
                device=world_se3_camera.device,
                dtype=world_se3_camera.dtype
            )

        assert limits.shape[-1] == 4, \
            ("Limits tensor must be of [..., 4] shape: "
             "phi_min, phi_range, theta_min, theta_range")
        assert (limits[..., 1] > 0).all(), "phi_range must be positive"
        self.limits = limits

    @property
    def range(self):
        return self.limits[..., [1, 3]]

    @property
    def offsets(self):
        return self.limits[..., [0, 2]]

    @property
    def phi_min(self):
        return self.limits[..., 0]

    @property
    def phi_range(self):
        return self.limits[..., 1]

    @property
    def phi_max(self):
        return self.phi_min + self.phi_range

    @property
    def theta_min(self):
        return self.limits[..., 2]

    @property
    def theta_range(self):
        return self.limits[..., 3]

    @property
    def theta_max(self):
        return self.theta_min + self.theta_range

    def uv_to_camera(self, uv: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        return spherical_to_cartesian(torch.cat([
            # uv to [azimuth,elevation] in radians
            uv * self.range + self.offsets,
            depth.unsqueeze(-1)
        ], dim=-1))

    def camera_to_uv(self, points: torch.Tensor) -> torch.Tensor:
        spherical = cartesian_to_spherical(points)
        angles, dist = spherical[..., :2], spherical[..., 2:3]
        uv = (angles - self.offsets) / self.range
        return torch.cat([uv, dist], dim=-1)
