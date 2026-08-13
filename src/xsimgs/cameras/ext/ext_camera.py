from contextlib import contextmanager

import torch

from xsimgs._C import isect_regular_tiles


class FusedRSCamera:
    ext_cls = None

    use_phm: bool = False

    _pack: torch.Tensor | None = None
    _freeze_flag: bool = False

    @property
    def _pack_data(self) -> list[torch.Tensor]:
        return [
            self.world_se3_camera.pose.q.data.view(4).contiguous(),
            self.world_se3_camera.pose.t.view(3).contiguous(),
            self.world_se3_camera.linear_velocity.view(3).contiguous(),
            self.world_se3_camera.angular_velocity.view(3).contiguous(),
            self.shutter.duration.view(2).contiguous(),
            self.shutter.offset_time.view(1),
            self.world_se3_camera.pose_time.view(1)
        ]

    def _set_freeze(self, value: bool):
        self._freeze_flag = value
        self._pack = None

    @contextmanager
    def _freeze_ctx(self):
        needed = not self._freeze_flag
        if needed:
            self._set_freeze(True)
        try:
            yield
        finally:
            if needed:
                self._set_freeze(False)
                
    @classmethod
    def supports_phm(cls) -> bool:
        return hasattr(cls.ext_cls, 'projection_utgs_phm')

    @property
    def pack(self):
        if self._freeze_flag and self._pack is not None:
            return self._pack
        self._pack = self.ext_cls.pack(self._pack_data)
        return self._pack

    def world_to_uv(
            self,
            points: torch.Tensor,
            points_velocity: torch.Tensor = None,
            points_time: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if points_time is not None and points_velocity is not None:
            # Transform points to world_se3_camera.pose_time timestamp
            delta_t = points_time - self.world_se3_camera.pose_time
            points = points + delta_t * points_velocity

        assert points.device == self.device, \
            ("Points and camera must be on the same device."
             "Points: {}, camera: {}".format(points.device, self.device))

        return self.ext_cls.projection(
            self.pack,
            points.contiguous(),
            points_velocity.contiguous() if points_velocity is not None else None,
            residual_thr_early_stop=self.residual_thr_early_stop,
            max_residual=self.residual_max_valid,
            max_margin=self.max_margin
        )

    def camera_rays(
            self,
            width: int,
            height: int,
            normalized_rays: bool = True,
            world_space: bool = True,
            return_origins: bool = True,
            return_valid: bool = False
    ) -> tuple[torch.Tensor | None, torch.Tensor] | \
         tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        out = self.ext_cls.camera_rays(
            self.pack,
            width=width,
            height=height,
            world_space=world_space,
            normalized=normalized_rays,
            return_valid=return_valid,
        )
        ray_o = out[0] if return_origins else None
        if return_valid:
            return ray_o, out[1], out[2]
        return ray_o, out[1]

    def uv_time(self, uv: torch.Tensor) -> torch.Tensor:
        return self.ext_cls.pixel_time_from_uv(
            self.pack, uv.contiguous()
        ).unsqueeze(-1)

    def uv_time_delta(self, uv: torch.Tensor) -> torch.Tensor:
        return self.ext_cls.pixel_time_delta_from_uv(
            self.pack, uv.contiguous()
        ).unsqueeze(-1)

    def image_time(self, image: torch.Tensor) -> torch.Tensor:
        return self.ext_cls.pixel_time(
            self.pack, width=image.shape[1], height=image.shape[0]
        ).unsqueeze(-1)

    def image_time_delta(self, image: torch.Tensor) -> torch.Tensor:
        return self.ext_cls.pixel_time_delta(
            self.pack, width=image.shape[1], height=image.shape[0]
        ).unsqueeze(-1)

    def undistort_map(
            self,
            width: int,
            height: int,
            normalized_range: bool = False
    ):
        if not hasattr(self.ext_cls, 'undistort_map'):
            raise NotImplementedError(
                'Camera C++/CUDA extension class "{}"'
                'does not implement "undistort_map"'.format(self.ext_cls.__name__))
        return self.ext_cls.undistort_map(
            self.pack,
            width=width, height=height,
            normalized_range=normalized_range
        )

    def ut_gs_projection(
            self,
            points: torch.Tensor,
            scale: torch.Tensor,
            rotation: torch.Tensor,
            width: int,
            height: int,
            points_velocity: torch.Tensor = None,
            opacity: torch.Tensor = None,
            ut_alpha: float = 0.1,
            ut_beta: float = 2.0,
            ut_kappa: float = 0.0,
            use_phm: bool = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Returns flag, radii, mean2d, depth
        if use_phm is None:
            use_phm = self.use_phm
        # Phase modeling where the camera model provides it, the plain
        # projection where it does not: a request is a preference, not a
        # requirement, so one flag can cover a pipeline of mixed camera models.
        proj_fn = (
            self.ext_cls.projection_utgs_phm
            if use_phm and self.supports_phm()
            else self.ext_cls.projection_utgs
        )
        return proj_fn(
            self.pack,
            points.contiguous(),
            points_velocity.contiguous() if points_velocity is not None else None,
            scale.contiguous(),
            rotation.contiguous(),
            opacity.contiguous() if opacity is not None else None,
            width=width,
            height=height,
            ut_alpha=ut_alpha,
            ut_beta=ut_beta,
            ut_kappa=ut_kappa,
            residual_thr_early_stop=self.residual_thr_early_stop,
            max_residual=self.residual_max_valid,
            max_margin=self.max_margin
        )

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
        return isect_regular_tiles(
            means2d, radii, depths,
            img_width=width, img_height=height,
            tile_size_x=tile_size_x, tile_size_y=tile_size_y,
            sort=sort, double_buffer=double_buffer
        )
