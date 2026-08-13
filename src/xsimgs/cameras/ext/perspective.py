import torch

from xsimgs._C import PerspectiveCamera as CPerspectiveCamera

from ..torch import PerspectiveCameraTorch
from .ext_camera import FusedRSCamera


class PerspectiveCamera(FusedRSCamera, PerspectiveCameraTorch):
    ext_cls = CPerspectiveCamera

    @property
    def _pack_data(self) -> list[torch.Tensor]:
        return super()._pack_data + [
            self.normalized_intrinsics.view(4).contiguous()
        ]
