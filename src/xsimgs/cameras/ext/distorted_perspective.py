import torch

from xsimgs._C import DistortedPerspectiveCamera as CDistortedPerspectiveCamera

from ..torch import DistortedPerspectiveCameraTorch
from .ext_camera import FusedRSCamera

from .perspective import PerspectiveCamera

class DistortedPerspectiveCamera(FusedRSCamera, DistortedPerspectiveCameraTorch):
    undistorted_cls = PerspectiveCamera
    ext_cls = CDistortedPerspectiveCamera

    @property
    def _pack_data(self) -> list[torch.Tensor]:
        return super()._pack_data + [
            self.normalized_intrinsics.view(4).contiguous(),
            self.distortion.view(8).contiguous(),
            # Squared, so the projection tests it without taking a root.
            self.max_radius.reshape(1).square().contiguous()
        ]
