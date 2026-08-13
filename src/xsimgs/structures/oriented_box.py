from pathlib import Path
from typing import Optional

import torch

from toast.utils.data_tensor import DataTensor
from toast import SE3, LinearMotion

from xsimgs._C import point_boxes_intersection

from .box import BOX_VERTICES, BOX_EDGES

def _as_tensor(v, ref):
    if isinstance(v, float):
        v = [v, v, v]
    if isinstance(v, list):
        v = torch.as_tensor(v)
    if isinstance(v, torch.Tensor):
        v = v.to(device=ref.device, dtype=ref.dtype)
        assert v.shape[-1] == 3
    return v


class BoxSensorLink(DataTensor):
    _buffers = ["sensor_type", "sensor_id", "sample_idx"]

    def __init__(
            self,
            sensor_type: torch.Tensor,
            sensor_id: torch.Tensor,
            sample_idx: torch.Tensor
    ):
        assert sensor_type.shape == sensor_id.shape == sample_idx.shape,\
            "sensor_type, sensor_id, sample_idx shapes must match. Given: {}, {}, {}".format(
                list(sensor_type.shape), list(sensor_id.shape), list(sample_idx.shape)
            )
        assert sensor_type.shape[-1] == 1, "Tensors must be of [..., 1] shape"
        self.sensor_type = sensor_type
        self.sensor_id = sensor_id
        self.sample_idx = sample_idx

    @staticmethod
    def from_indices(sensor_type: int, indices: list[tuple[int, int]]):
        type_t = (torch.tensor(sensor_type, dtype=torch.int32)
                  .expand(len(indices))
                  .unsqueeze(-1))
        sensor_ids = torch.tensor([x[0] for x in indices], dtype=torch.int32).view(-1, 1)
        sample_idx = torch.tensor([x[1] for x in indices], dtype=torch.int32).view(-1, 1)

        return BoxSensorLink(
            sensor_type=type_t,
            sensor_id=sensor_ids,
            sample_idx=sample_idx
        )

    def key(self) -> torch.Tensor:
        s_type = (self.sensor_type & 0xff).long().bitwise_left_shift(24)
        s_id = (self.sensor_id & 0xffffff).long() + s_type
        return s_id.bitwise_left_shift(32) + self.sample_idx.long()


class OrientedBoxes(DataTensor):
    _buffers = ["motion", "sizes"]

    _optional_buffers = [
        "category",
        "object_ids",
        "mask",
        "sensor_link"
    ]

    motion: LinearMotion
    sizes: torch.Tensor
    category: Optional[torch.Tensor]
    object_ids: Optional[torch.Tensor]
    mask: Optional[torch.Tensor]
    sensor_link: Optional[BoxSensorLink]

    def __init__(
            self,
            motion: LinearMotion,
            sizes: torch.Tensor,
            category: Optional[torch.Tensor] = None,
            object_ids: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            sensor_link: Optional[BoxSensorLink] = None,
    ):
        self.motion = motion
        self.sizes = sizes
        self.category = category
        self.object_ids = object_ids
        self.mask = mask
        self.sensor_link = sensor_link

    @property
    def pose(self) -> SE3:
        return self.motion.pose

    @staticmethod
    def zeros(size: list[int], device: str = 'cpu'):
        return OrientedBoxes(
            motion=LinearMotion(
                pose=SE3.unit(*size, device=device),
                pose_time=torch.zeros(*size, 1, device=device),
                linear_velocity=torch.zeros(*size, 3, device=device),
                angular_velocity=torch.zeros(*size, 3, device=device)
            ),
            sizes=torch.zeros(*size, 3, device=device),
            category=torch.zeros(*size, device=device, dtype=torch.int32),
            object_ids=torch.zeros(*size, device=device, dtype=torch.int32),
            mask=torch.zeros(*size, device=device, dtype=torch.bool),
            sensor_link=BoxSensorLink(
                sensor_type=torch.zeros(size=(*size, 1), device=device, dtype=torch.int32),
                sensor_id=torch.zeros(size=(*size, 1), device=device, dtype=torch.int32),
                sample_idx=torch.zeros(size=(*size, 1), device=device, dtype=torch.int32),
            )
        )

    @staticmethod
    def empty():
        return OrientedBoxes(
            motion=LinearMotion(
                pose=SE3.unit(0),
                pose_time=torch.zeros(0, 1),
                linear_velocity=torch.zeros(0, 3),
                angular_velocity=torch.zeros(0, 3)
            ),
            sizes=torch.zeros(0, 3)
        )

    @property
    def centers(self) -> torch.Tensor:
        return self.motion.pose.t

    @property
    def quats(self) -> torch.Tensor:
        return self.motion.pose.q.data

    @property
    def half_sizes(self) -> torch.Tensor:
        return 0.5 * self.sizes

    @property
    def bev_area(self) -> torch.Tensor:
        return self.sizes[..., 0] * self.sizes[..., 1]

    @property
    def volume(self) -> torch.Tensor:
        return self.sizes.prod(dim=-1)

    def transform(self, se3: SE3) -> "OrientedBoxes":
        result = self.clone()
        result.motion = self.motion.transform(se3.broadcast_to(self.motion.shape))
        return result

    def scale(self, scale: torch.Tensor | list[float] | float):
        result = self.clone()
        result.sizes *= _as_tensor(scale, self.sizes).view(-1, 3)
        return result

    def translate(self, t: torch.Tensor | list[float] | float):
        result = self.clone()
        result.sizes += _as_tensor(t, self.sizes).view(-1, 3)
        return result

    def intersection_map(self, points: torch.Tensor) -> torch.Tensor:
        """
        Given points [N, 3] tensor, and K oriented boxes, returns
        index of oriented box which point belongs to. If point does not
        belong to any oriented box, returns -1 for this point.
        Returns int32 tensor of shape [N, ]
        """
        assert points.device == self.device,\
            ("Points must be on the same device as OrientedBoxes. "
             "Points: {} Boxes: {}").format(points.device, self.device)

        return point_boxes_intersection(
            points.contiguous().view(-1, 3),
            self.motion.pose.q.data.contiguous().view(-1, 4),
            self.motion.pose.t.contiguous().view(-1, 3),
            self.sizes.contiguous().view(-1, 3)
        )

    def semantic_map(self, points: torch.Tensor) -> torch.Tensor:
        assert self.category is not None, "Category must be set for bounding boxes"
        # 0 = nothing, 1 = category#0, ...
        box_idxs = self.intersection_map(points) + 1
        box_idx_to_cat = torch.zeros(
            len(self.category) + 1,
            dtype=self.category.dtype,
            device=self.category.device
        )
        box_idx_to_cat[1:] = self.category + 1
        return box_idx_to_cat[box_idxs]

    def are_points_inside(self, points: torch.Tensor) -> torch.Tensor:
        return self.intersection_map(points) != -1

    def are_points_outside(self, points: torch.Tensor) -> torch.Tensor:
        return self.intersection_map(points) == -1

    def vertices(self):
        vertices = BOX_VERTICES.view(1, 8, 3).to(device=self.device, dtype=self.dtype)
        vertices = self.half_sizes.view(-1, 1, 3) * vertices.expand(len(self), -1, -1)
        return self.motion.pose.view(-1, 1) @ vertices

    def vis_mesh(self):
        from plytorch import Lines
        vertices = self.vertices()
        inst_offsets = torch.arange(len(self), dtype=torch.int) * 8
        inst_offsets = inst_offsets.repeat_interleave(len(BOX_EDGES)).view(-1, 1)
        edges = BOX_EDGES.repeat(len(self), 1) + inst_offsets
        return Lines(points=vertices.view(-1, 3), edges=edges)

    def filter_by_roi(self, roi, allow_partial_objects: bool = True):
        mask = roi.is_inside(self.vertices().view(-1, 3)).view(len(self), 8)
        mask = mask.any(dim=1) if allow_partial_objects else mask.all(dim=1)
        return self[mask]

    def at_time(self, query_t):
        """Move every box along its own velocity to a queried time.

        Args:
            query_t: Time to pose the boxes at. A scalar applies to all of them,
                otherwise it broadcasts over the batch, one time per box.

        Returns:
            A copy posed at ``query_t``, velocities unchanged.
        """
        if not isinstance(query_t, torch.Tensor):
            query_t = torch.tensor(query_t, device=self.device)

        query_t = query_t.broadcast_to(self.motion.pose_time.shape)
        poses_at_query = self.motion.pose_at(query_t)

        result = self.clone()
        result.motion.pose = poses_at_query
        result.motion.pose_time = query_t
        return result

    def sort_by_time(self, descending: bool = False) -> "OrientedBoxes":
        return self[self.motion.pose_time.flatten().argsort(descending=descending)]

    def sample_group_iterator(self):
        keys = self.sensor_link.key().flatten()
        keys, key_indices = keys.sort(stable=True)
        _, counts = torch.unique_consecutive(keys, dim=0, return_counts=True)
        counts = counts.cpu()

        start_idx = 0
        for cur_count in counts:
            cur_count = int(cur_count)
            cur_indices = key_indices[start_idx: start_idx + cur_count]
            yield (cur_indices, self[cur_indices])
            start_idx += cur_count
