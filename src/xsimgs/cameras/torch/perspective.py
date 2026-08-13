import torch
from toast import LinearMotion

from xsimgs.cameras.base.rs_camera import RSCamera, CameraNameType
from xsimgs.cameras.base.shutter import RollingShutter


class PerspectiveCameraTorch(RSCamera):
    _buffers = RSCamera._buffers + ["normalized_intrinsics"]

    # Tensor with [fx, fy, cx, cy] normalized by image size
    normalized_intrinsics: torch.Tensor

    def __init__(
            self,
            world_se3_camera: LinearMotion,
            shutter: RollingShutter,
            normalized_intrinsics: torch.Tensor,
            camera_name: CameraNameType = None
    ):
        super().__init__(
            world_se3_camera=world_se3_camera,
            shutter=shutter,
            camera_name=camera_name
        )
        self.normalized_intrinsics = normalized_intrinsics

    @staticmethod
    def intrinsics_from_sized(
            fx: float,
            fy: float,
            cx: float,
            cy: float,
            width: int,
            height: int
    ) -> torch.Tensor:
        return torch.tensor(
            [fx/width, fy/height, cx/width, cy/height],
            dtype=torch.float32
        )

    @property
    def f(self):
        return self.normalized_intrinsics[..., :2]

    @property
    def fx(self):
        return self.normalized_intrinsics[..., 0]

    @property
    def fy(self):
        return self.normalized_intrinsics[..., 1]

    @property
    def c(self):
        return self.normalized_intrinsics[..., 2:]

    @property
    def cx(self):
        return self.normalized_intrinsics[..., 2]

    @property
    def cy(self):
        return self.normalized_intrinsics[..., 3]

    def uv_to_camera(self, uv: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        if depth.ndim == 2:
            depth = depth.unsqueeze(-1)

        return torch.cat([
            (uv - self.c) / self.f,
            torch.ones_like(uv[..., 0:1])
        ], dim=-1) * depth

    def camera_to_uv(self, points: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            (points[..., :2] / points[..., 2:3]) * self.f + self.c,
            points[..., 2:3]
        ], dim=-1)

    def sized_intrinsics(self, width: int, height: int) -> torch.Tensor:
        """Returns sized intrinsics from this camera normalized intrinsics"""
        norm_k = torch.as_tensor([width, height, width, height])
        norm_k = norm_k.float().to(self.normalized_intrinsics.device)
        return self.normalized_intrinsics * norm_k

    def k_matrix(self, width: int = 1, height: int = 1):
        fx, fy, cx, cy = self.sized_intrinsics(width, height)
        K = torch.eye(3, device=fx.device)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        return K
