import torch
from toast import LinearMotion

from .distortion_utils import iterate_undistortion, max_valid_radius, uv_distortion
from .perspective import PerspectiveCameraTorch
from .. import RollingShutter
from ..base.rs_camera import CameraNameType


class DistortedPerspectiveCameraTorch(PerspectiveCameraTorch):
    undistorted_cls = PerspectiveCameraTorch
    _buffers = PerspectiveCameraTorch._buffers + ["distortion"]

    # Eight coefficient tensor: k1, k2, p1, p2, k3, k4, k5, k6
    # k1...k6 - radial distortion
    # p1, p2 - tangential distortion
    # For more info refer to https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
    distortion: torch.Tensor

    # How far past the calibrated image corner the model is still trusted; see
    # max_valid_radius, which also caps this at the radius where the calibration
    # folds. Raise it to keep gaussians further outside the frame, whose splats
    # can still reach into it.
    max_radius_margin: float = 1.5

    def check_distortion(self):
        assert self.distortion.shape[-1] == 8, \
            ("Distortion coefficients must be of [..., 8] shape: "
             "k1, k2, p1, p2, k3, k4, k5, k6")

    @property
    def max_radius(self) -> torch.Tensor:
        """Radius bound for this calibration, as a [..., 1] tensor.

        Derived from the coefficients and intrinsics rather than stored, so it
        cannot fall out of step with them, and cached because the derivation
        scans for the fold. Slicing or moving the camera builds a new instance,
        which recomputes on first use.
        """
        if getattr(self, '_max_radius', None) is None:
            self._max_radius = max_valid_radius(
                self.normalized_intrinsics,
                self.distortion,
                margin=self.max_radius_margin
            )

        return self._max_radius

    def __init__(
            self,
            world_se3_camera: LinearMotion,
            shutter: RollingShutter,
            normalized_intrinsics: torch.Tensor,
            distortion: torch.Tensor,
            camera_name: CameraNameType = None
    ):

        super().__init__(
            world_se3_camera=world_se3_camera,
            shutter=shutter,
            normalized_intrinsics=normalized_intrinsics,
            camera_name=camera_name
        )
        self.distortion = distortion
        self.check_distortion()

    def uv_to_camera(self, uv: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        if depth.ndim == 2:
            depth = depth.unsqueeze(-1)

        uv_cam = iterate_undistortion((uv - self.c) / self.f, self.distortion)
        return torch.cat([
            uv_cam, torch.ones_like(uv_cam[..., 0:1])
        ], dim=-1) * depth

    def camera_to_uv(self, points: torch.Tensor) -> torch.Tensor:
        depth = points[..., 2:3]
        xy_ = points[..., :2] / depth
        radial_scale, tang_offset = uv_distortion(xy_, self.distortion)

        uv = (xy_ * radial_scale + tang_offset) * self.f + self.c
        return torch.cat([uv, depth], dim=-1)

    def undistort(self):
        return self.undistorted_cls(
            world_se3_camera=self.world_se3_camera,
            shutter=self.shutter,
            normalized_intrinsics=self.normalized_intrinsics,
            camera_name=self.camera_name
        )
