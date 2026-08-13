import torch

from xsimgs._C import OpenCVFisheyeCamera as CFisheyeCamera

from ..torch import FisheyeCameraTorch
from .ext_camera import FusedRSCamera


class FisheyeCamera(FusedRSCamera, FisheyeCameraTorch):
    ext_cls = CFisheyeCamera

    @property
    def _pack_data(self) -> list[torch.Tensor]:
        return super()._pack_data + [
            self.normalized_intrinsics.view(4).contiguous(),
            self.distortion.view(4).contiguous()
        ]

