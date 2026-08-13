import enum
from typing import Optional

import torch

from toast import DataTensor

from .utils import uv_grid, uv_grid_like


class RollingShutterDirection(enum.Enum):
    # All pixels are captured simultaneously
    GLOBAL_SHUTTER = 0

    # Column-by-column readout. x=0 captured first
    LEFT_TO_RIGHT = 1

    # Column-by-column readout. x=width-1 captured first
    RIGHT_TO_LEFT = 2

    # Row-by-row readout. Top row first, bottom row last. y=0 captured first
    TOP_TO_BOTTOM = 3

    # Row-by-row readout. Bottom row first, top row last. y=height-1 captured first
    BOTTOM_TO_TOP = 4


class RollingShutter(DataTensor):
    _buffers = ["offset_time", "duration"]
    _optional_buffers = ["exposure_time"]

    # Tensor [..., 1], time when pixel [0,0] was captured
    offset_time: torch.Tensor
    # Tensor [..., 2] - X and Y rolling shutter capture duration
    duration: torch.Tensor
    # Duration of exposure
    exposure_time: torch.Tensor | None = None

    def __init__(
            self,
            offset_time: torch.Tensor,
            duration: torch.Tensor,
            exposure_time: torch.Tensor = None
    ):
        assert offset_time.shape[-1] == 1, "Offset time must be [..., 1] tensor"
        assert duration.shape[-1] == 2, "Duration time must be [..., 2] tensor"
        assert (exposure_time is None) or (exposure_time.shape[-1] == 1), \
            "If provided, exposure time must be [..., 1] tensor"
        self.offset_time = offset_time
        self.duration = duration
        self.exposure_time = exposure_time

    @staticmethod
    def from_type(
            trigger_time: float | torch.Tensor,
            duration: float | torch.Tensor,
            rs_type: RollingShutterDirection,
            exposure_time: Optional[torch.Tensor] = None
    ) -> "RollingShutter":
        t = torch.as_tensor(trigger_time)
        d = torch.as_tensor(duration)

        assert t.shape[-1] == 1, "Trigger time must be [..., 1] tensor"
        assert d.shape[-1] == 1, "Duration time must be [..., 1] tensor"

        duration = torch.zeros(
            list(t.shape[:-1]) + [2,], device=t.device, dtype=t.dtype)

        rs_type_value = rs_type.value
        if rs_type_value == RollingShutterDirection.GLOBAL_SHUTTER.value:
            return RollingShutter(t, duration)

        positive = rs_type_value in [
            RollingShutterDirection.LEFT_TO_RIGHT.value,
            RollingShutterDirection.TOP_TO_BOTTOM.value
        ]
        horizontal = rs_type_value in [
            RollingShutterDirection.LEFT_TO_RIGHT.value,
            RollingShutterDirection.RIGHT_TO_LEFT.value
        ]
        duration[..., 0 if horizontal else 1] = (d if positive else -d)[..., 0]
        offset_time = t if positive else t + d
        return RollingShutter(offset_time, duration, exposure_time=exposure_time)

    @staticmethod
    def from_time_image(
            time_image: torch.Tensor,
            mask: torch.Tensor = None,
            horizontal_only: bool = False
    ) -> "RollingShutter":
        time_image = time_image.squeeze(-1)
        if mask is not None:
            mask = mask.squeeze(-1)

        uv1 = uv_grid_like(time_image, with_ones=True)
        if mask is not None:
            uv1 = uv1[mask]
            time_image = time_image[mask]
        else:
            uv1 = uv1.view(-1, 3)
            time_image = time_image.view(-1, 1)

        if horizontal_only:
            uv1 = uv1[:, [0,2]]

        solution = torch.linalg.lstsq(uv1, time_image).solution.view(-1)
        time_offset = solution[-1].view(1)
        duration = solution[:-1].view(-1)
        if horizontal_only:
            duration = torch.cat(
                [duration, torch.zeros_like(duration)], dim=0)
        return RollingShutter(time_offset, duration)


    @property
    def direction(self) -> RollingShutterDirection:
        assert self.numel == 1, "Direction can not be called on batched shutter"
        x_dur, y_dur = self.duration.view(2).tolist()
        if x_dur == 0 and y_dur == 0:
            return RollingShutterDirection.GLOBAL_SHUTTER

        if abs(x_dur) > abs(y_dur):
            return (
                RollingShutterDirection.LEFT_TO_RIGHT
                if x_dur > 0 else
                RollingShutterDirection.RIGHT_TO_LEFT
            )
        else:
            return (
                RollingShutterDirection.TOP_TO_BOTTOM
                if y_dur > 0 else
                RollingShutterDirection.BOTTOM_TO_TOP
            )

    @property
    def x_duration(self) -> torch.Tensor:
        return self.duration[..., 0:1]

    @property
    def y_duration(self) -> torch.Tensor:
        return self.duration[..., 1:2]

    @property
    def time_range(self) -> tuple[torch.Tensor, torch.Tensor]:
        t1 = self.offset_time
        t2 = self.offset_time + self.x_duration
        t3 = self.offset_time + self.y_duration
        tmin = torch.minimum(torch.minimum(t1, t2), t3)
        tmax = torch.maximum(torch.maximum(t1, t2), t3)
        return tmin, tmax

    def uv_time(self, uv: torch.Tensor) -> torch.Tensor:
        assert uv.shape[-1] == 2, "UV must be [..., 2] tensor"
        assert uv.device == self.device, "UV must be on the same device"
        dot = (self.duration.unsqueeze(-2) * uv).sum(dim=-1, keepdim=True)
        return self.offset_time.unsqueeze(-2) + dot

    def image_time(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[:2]
        uv = uv_grid(
            width=width, height=height,
            device=image.device, dtype=self.dtype
        )
        return self.uv_time(uv)

    def lerp(
            self,
            other: "RollingShutter",
            t: float | torch.Tensor
    ) -> "RollingShutter":
        t = torch.as_tensor(t)

        return RollingShutter(
            torch.lerp(self.offset_time, other.offset_time, t),
            torch.lerp(self.duration, other.duration, t),
            exposure_time=torch.lerp(self.exposure_time, other.exposure_time, t)
                if (self.exposure_time is not None)
                   and (other.exposure_time is not None)
                else None
        )

