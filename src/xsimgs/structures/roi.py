import numpy as np
import torch
from toast.ops.ext import last_dim_ctg

from xsimgs._C import vg_is_inside

from .box import BOX_VERTICES, BOX_EDGES

class ROIBox3D:
    """
    Axis-aligned bounding box that represents region of interest

    Attributes
    ----------
    vmin: torch.Tensor
        Minimum (x, y, z) coordinates of ROI
    vmax: torch.Tensor
        Maximum (x, y, z) coordinates of ROI
    """

    vmin: torch.Tensor
    vmax: torch.Tensor

    def __repr__(self):
        return "{}(vmin={}, vmax={})".format(
            self.__class__.__name__,
            self.vmin.cpu().tolist(),
            self.vmax.cpu().tolist()
        )

    def intersection(self, other):
        return ROIBox3D(
            vmin=torch.maximum(self.vmin, other.vmin),
            vmax=torch.minimum(self.vmax, other.vmax)
        )

    def union(self, other):
        return ROIBox3D(
            vmin=torch.minimum(self.vmin, other.vmin),
            vmax=torch.maximum(self.vmax, other.vmax)
        )

    def __and__(self, other):
        return self.intersection(other)

    def __or__(self, other):
        return self.union(other)

    @staticmethod
    def from_points(points: torch.Tensor):
        assert points.ndim == 2 and points.shape[-1] == 3
        return ROIBox3D(
            vmin=points.amin(dim=0).cpu(),
            vmax=points.amax(dim=0).cpu()
        )

    def is_equal(self, other):
        if other is None:
            return False

        return (torch.allclose(self.vmin, other.vmin)
                and torch.allclose(self.vmax, other.vmax))

    def __init__(self,
                 vmin: torch.Tensor | list[float],
                 vmax: torch.Tensor | list[float]):
        super().__init__()

        if not isinstance(vmin, torch.Tensor):
            vmin = torch.as_tensor(vmin).float()
        if not isinstance(vmax, torch.Tensor):
            vmax = torch.as_tensor(vmax).float()
        assert vmin.numel() == 3 and vmax.numel() == 3

        self.vmin = vmin
        self.vmax = vmax

    @staticmethod
    def from_values(
            xmin: float = -torch.inf,
            xmax: float = torch.inf,
            ymin: float = -torch.inf,
            ymax: float = torch.inf,
            zmin: float = -torch.inf,
            zmax: float = torch.inf
    ):
        return ROIBox3D(
            torch.as_tensor([xmin, ymin, zmin]),
            torch.as_tensor([xmax, ymax, zmax])
        )

    def __class_getitem__(cls, item):
        """
        Construct ROIBox3D by slicing class name:
        E.g.: ROIBox3D[-100:150, -100:100] constructs ROI
        with x \in [-100; 150] range,
             y \in [-100; 100] range,
             z \in [-\infty; \infty] range
        """
        item = [item] if isinstance(item, slice) else item
        assert len(item) <= 3
        to_limits = lambda x: (
            x.start if x.start is not None else -np.inf,
            x.stop if x.stop is not None else np.inf
        )
        limits = torch.as_tensor(
            list(map(to_limits, list(item) +
                     [slice(None, None, None)] * (3 - len(item)))))
        return ROIBox3D(vmin=limits[:, 0], vmax=limits[:, 1])

    @property
    def center(self) -> torch.Tensor:
        return (self.vmin + self.vmax) * 0.5

    @property
    def size(self) -> torch.Tensor:
        """
        Metric size of ROI

        Returns
        -------
        torch.Tensor:
            Tensor with shape [3, ]
        """
        return self.vmax - self.vmin

    def voxel_grid_size(self, voxel_size: float | torch.Tensor, round_up: bool = True):
        size = (self.vmax - self.vmin) / voxel_size
        if round_up:
            size = torch.ceil(size)
        return size.int()

    def normalize_into_roi(
            self,
            points: torch.Tensor,
            clamp: bool = True
    ) -> torch.Tensor:
        """
        Normalizes points from metric coordinate space into [0; 1] space
        formed by ROIBox3D boundaries

        Parameters
        ----------
        points: torch.Tensor
            Tensor of shape (..., 3). Points to transform

        clamp: bool
            If true, point outside of ROI clamped to [0; 1].
            Otherwise, points outside ROI will exceed [0; 1] range

        Returns
        -------
        torch.Tensor
            Points normalized into [0; 1] range
        """
        result = (points - self.vmin.to(points.device)) / self.size.to(points.device)
        if clamp:
            result = result.clamp(0, 1)
        return result

    def normalized_roi_into_world(self, points: torch.Tensor) -> torch.Tensor:
        return points * self.size.to(points.device) + self.vmin.to(points.device)

    def check_inside(self, points: torch.Tensor) -> torch.Tensor:
        """
        Checks if points are inside the ROIBox3D.

        Parameters
        ----------
        points: torch.Tensor
            Tensor of shape (..., 3). Points to check.

        Returns
        -------
        torch.Tensor:
            Boolean tensor of shape (..., 1) indicating if points are inside the ROIBox3D.
        """
        assert points.shape[-1] == 3
        return vg_is_inside(self.vmin, self.vmax, last_dim_ctg(points))

    def check_outside(self, points: torch.Tensor) -> torch.Tensor:
        """
        Checks if points are outside the ROIBox3D.

        Parameters
        ----------
        points: torch.Tensor
            Tensor of shape (..., 3). Points to check.

        Returns
        -------
        torch.Tensor:
            Boolean tensor of shape (...) indicating if points are outside the ROIBox3D.
        """
        return ~self.check_inside(points)

    def intersect_with_rays(
            self,
            ray_origins: torch.Tensor,
            ray_dirs: torch.Tensor
    ):
        t_min = (self.vmin.to(ray_origins.device) - ray_origins) / ray_dirs
        t_max = (self.vmax.to(ray_origins.device) - ray_origins) / ray_dirs
        t1 = torch.min(t_min, t_max)
        t2 = torch.max(t_min, t_max)
        t_near = torch.max(torch.max(t1[..., 0], t1[..., 1]), t1[..., 2])
        t_far = torch.min(torch.min(t2[..., 0], t2[..., 1]), t2[..., 2])
        return t_near, t_far

    def vis_mesh(self):
        from plytorch import Lines

        use_vmax = BOX_VERTICES > 0
        vmax = self.vmax.view(1, 3).expand(len(BOX_VERTICES), -1)
        vmin = self.vmin.view(1, 3).expand_as(vmax)

        vertices = torch.where(use_vmax, vmax, vmin)
        edges = BOX_EDGES
        return Lines(points=vertices, edges=edges)


class ROIBox3DModule(torch.nn.Module, ROIBox3D):
    def __init__(self, vmin, vmax):
        ROIBox3D.__init__(self, vmin, vmax)
        torch.nn.Module.__init__(self)

        vmin = self.vmin.clone()
        vmax = self.vmax.clone()
        del self.vmin
        del self.vmax
        self.register_buffer('vmin', vmin)
        self.register_buffer('vmax', vmax)

    @staticmethod
    def from_roi(roi: ROIBox3D) -> 'ROIBox3DModule':
        return ROIBox3DModule(vmin=roi.vmin, vmax=roi.vmax)

    @staticmethod
    def from_state_dict(state):
        m = ROIBox3DModule(vmin=[0,0,0], vmax=[0,0,0])
        if ('vmin' in state) and ('vmax' in state):
            m.load_state_dict(state)
        return m

    def __repr__(self):
        return ROIBox3D.__repr__(self)

    def clone(self):
        return ROIBox3DModule(
            vmin=self.vmin.detach().clone(),
            vmax=self.vmax.detach().clone()
        )

