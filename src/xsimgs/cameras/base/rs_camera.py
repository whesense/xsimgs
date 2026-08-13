from typing import Union, Optional

import torch
import torch.nn.functional as F

from toast import LinearMotion, SE3, DataTensor

from .shutter import RollingShutter
from .utils import rasterize_into_depth_mask, pixel_grid


CameraNameType = Optional[Union[str, int]]


class RSCamera(DataTensor):
    _buffers = ["world_se3_camera", "shutter"]
    _non_tensor_data = ["camera_name"]

    motion: LinearMotion
    shutter: RollingShutter
    camera_name: CameraNameType = None

    N_ROLLING_ITERATIONS: int = 5
    residual_thr_early_stop: float = 1e-4
    residual_max_valid: float = 0.002
    max_margin: float = 1.0

    def __init__(
            self,
            world_se3_camera: LinearMotion,
            shutter: RollingShutter,
            camera_name: CameraNameType = None
    ):
        self.world_se3_camera = world_se3_camera
        self.shutter = shutter
        self.camera_name = camera_name

    def uv_to_pixel(self, uv: torch.Tensor, width: int, height: int):
        assert uv.shape[-1] == 2
        size = torch.as_tensor(
            [width, height], dtype=uv.dtype, device=uv.device)
        return (uv * size - 0.5).round().int()

    def pixel_to_uv(self, xy: torch.Tensor, width: int, height: int):
        size = torch.tensor([width, height], device=xy.device)
        return (xy + 0.5) / size

    # Abstract methods
    def camera_to_uv(self, points: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def uv_to_camera(self, uv: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def camera_to_world(
            self,
            points: torch.Tensor,
            points_time: torch.Tensor
    ):
        return self.world_se3_camera.apply_to_points(
            points, points_time.squeeze(-1)
        )

    def world_to_uv(
            self,
            points: torch.Tensor,
            points_velocity: torch.Tensor = None,
            points_time: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:  # uvdt, mask
        if points_time is not None and points_velocity is not None:
            # Transform points to world_se3_camera.pose_time timestamp
            delta_t = points_time - self.world_se3_camera.pose_time
            points = points + delta_t * points_velocity

        valid_mask = torch.full_like(
            points[..., 0],
            fill_value=True,
            dtype=torch.bool
        )
        t_h = torch.zeros_like(points[..., 0])
        residual = 0

        for _ in range(self.N_ROLLING_ITERATIONS):
            cur_points = points
            if points_velocity is not None:
                cur_points = cur_points + t_h * points_velocity

            assumed_time = self.world_se3_camera.pose_time + t_h
            cam_poses = self.world_se3_camera.pose_at(assumed_time)
            points_cam = cam_poses.inverse_transform(cur_points)
            cur_uvd = self.camera_to_uv(points_cam)
            real_time = self.uv_time(cur_uvd[..., :2]).squeeze(-1)
            residual = assumed_time - real_time

            depth = cur_uvd[..., 2]
            cur_mask = (
                (depth > 0) &
                (cur_uvd[..., 0] >= -self.max_margin) &
                (cur_uvd[..., 0] <= 1.0 + self.max_margin) &
                (cur_uvd[..., 1] >= -self.max_margin) &
                (cur_uvd[..., 1] <= 1.0 + self.max_margin)
            )
            valid_mask &= cur_mask
            t_h[valid_mask] -= residual[valid_mask]

            if residual.abs().max() < self.residual_thr_early_stop:
                break

        valid_mask &= (residual.abs() < self.residual_max_valid)
        uvdt = torch.cat([cur_uvd, assumed_time.unsqueeze(-1)], dim=-1)
        return uvdt, valid_mask

    def uv_to_world(self, uv, depth) -> torch.Tensor:
        return self.camera_to_world(
            self.uv_to_camera(uv, depth),
            self.uv_time(uv)
        )

    def world_to_pixel(
            self,
            points: torch.Tensor,
            width: int,
            height: int,
            points_velocity: torch.Tensor = None,
            points_time: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        uvdt, mask = self.world_to_uv(
            points,
            points_velocity=points_velocity,
            points_time=points_time
        )
        xy = self.uv_to_pixel(uvdt[..., :2], width=width, height=height)
        xs, ys = xy.unbind(dim=-1)
        depth = uvdt[..., 2]
        mask = mask & (
                (ys >= 0) & (ys < height) &
                (xs >= 0) & (xs < width) &
                (depth > 0)
        )
        return xy, depth, mask

    def uv_grid(self, width: int, height: int):
        px_grid = pixel_grid(width=width, height=height, device=self.device)
        return self.pixel_to_uv(px_grid, width=width, height=height)

    def camera_rays(
            self,
            width: int,
            height: int,
            normalized_rays: bool = True,
            world_space: bool = True,
            return_origins: bool = True
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        device = self.device
        uv = self.uv_grid(width=width, height=height)
        ones = torch.ones(height, width, device=device)

        if world_space:
            p_ones = self.uv_to_world(uv, ones)
            zeros = torch.zeros(height, width, device=device)
            p_zeros = self.uv_to_world(uv, zeros)
            ray_o = p_zeros if return_origins else None
            ray_d = p_ones - p_zeros
        else:
            ray_d = self.uv_to_camera(uv, ones)
            ray_o = torch.zeros_like(ray_d) if return_origins else None

        if normalized_rays:
            ray_d = F.normalize(ray_d, dim=-1)

        return ray_o, ray_d

    def depth_to_camera(self, depth: torch.Tensor) -> torch.Tensor:
        assert depth.ndim == 2 or (depth.ndim == 3 and depth.shape[-1] == 1),\
            "Depth map must be [H,W] or [H,W,1] tensor"
        height, width = depth.shape[:2]
        depth = depth.view(height, width, 1)
        _, ray_d = self.camera_rays(
            width=width, height=height,
            normalized_rays=False,
            world_space=False,
            return_origins=False
        )
        return ray_d * depth

    def depth_to_world(self, depth: torch.Tensor) -> torch.Tensor:
        assert depth.ndim == 2 or (depth.ndim == 3 and depth.shape[-1] == 1), \
            "Depth map must be [H,W] or [H,W,1] tensor"
        height, width = depth.shape[:2]
        depth = depth.view(height, width, 1)
        ray_o, ray_d = self.camera_rays(
            width=width, height=height,
            normalized_rays=False,
            world_space=True,
            return_origins=True
        )
        return ray_o + ray_d * depth

    def transform(self, ref_se3_world: SE3):
        result: RSCamera = self.clone()
        result.world_se3_camera = self.world_se3_camera.transform(ref_se3_world)
        return result

    def lerp(self, other: "RSCamera", t: torch.Tensor):
        return RSCamera(
            world_se3_camera=self.world_se3_camera.lerp(other.world_se3_camera, t),
            shutter=self.shutter.lerp(other.shutter, t),
            camera_name=self.camera_name
        )

    @property
    def pose_range(self):
        ts = torch.stack(self.shutter.time_range).view(2)
        return self.world_se3_camera.pose_at(ts).to_matrix().unbind(dim=0)

    def uv_time(self, uv: torch.Tensor) -> torch.Tensor:
        return self.shutter.uv_time(uv)

    def uv_time_delta(self, uv: torch.Tensor) -> torch.Tensor:
        return self.uv_time(uv) - self.world_se3_camera.pose_time

    def image_time(self, image: torch.Tensor) -> torch.Tensor:
        return self.shutter.image_time(image)

    def image_time_delta(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_time(image) - self.world_se3_camera.pose_time

    def vis_mesh(self, size: float = 10.0):
        import plytorch

        uvs = torch.as_tensor([
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.5]
        ])
        depth = torch.as_tensor([size, size, size, size, 0.0])
        lines = torch.as_tensor([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 0],
            [4, 1],
            [4, 2],
            [4, 3]
        ]).int()

        device = self.device
        return plytorch.Lines(
            points=self.uv_to_world(uvs.to(device), depth.to(device)),
            edges=lines
        )



def rasterize_points_into_depth(
        camera: "RSCamera",
        points: torch.Tensor,
        width: int,
        height: int,
        points_velocity: torch.Tensor = None,
        points_time: torch.Tensor = None,
        near_plane: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    uvdt, mask = camera.world_to_uv(
        points,
        points_velocity=points_velocity,
        points_time=points_time
    )
    uvdt_masked = uvdt[mask]
    px, py = camera.uv_to_pixel(
        uvdt_masked[:, :2], width=width, height=height
    ).unbind(dim=-1)
    depth_map, rast_mask = rasterize_into_depth_mask(
        px, py, uvdt_masked[:, 2],
        width=width, height=height,
        near_plane=near_plane
    )
    mask[mask.clone()] = rast_mask
    return depth_map, mask

