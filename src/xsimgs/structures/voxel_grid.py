from pathlib import Path

import torch
from toast.ops.ext.ops_utils import last_dim_ctg

from .roi import ROIBox3D

from xsimgs._C import (
    vg_voxel_coord_flt,
    vg_voxel_coord_clamped,
    vg_voxel_coord_int,
    vg_voxel_offset,
    vg_is_inside
)


class VoxelGrid3D:
    roi: ROIBox3D
    voxel_size: torch.Tensor

    def __init__(self, roi: ROIBox3D, voxel_size: float | torch.Tensor):
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size]*3

        voxel_size = torch.as_tensor(voxel_size).cpu()
        self.roi = roi
        self.voxel_size = voxel_size

    @property
    def vmin(self) -> torch.Tensor:
        return self.roi.vmin

    @property
    def vmax(self):
        return self.roi.vmax

    @property
    def shape(self):
        """
        Number of voxels in grid given current vmin, vmax, and voxel_size

        Returns
        -------
        torch.Tensor
            CPU int32 tensor with (3, ) shape
        """
        return torch.ceil((self.vmax - self.vmin) / self.voxel_size).int()

    def points_into_voxel_clamped(self, points: torch.Tensor) -> torch.Tensor:
        """
        Return point coordinates in integer voxel grid coordinates
        (from 0 to self.grid_shape).
        If point is outside of voxel grid, coordinates are clamped
        into valid range.

        Parameters
        ----------
        points: torch.Tensor
            of [N, 3] shape. Points to be transformed

        Returns
        -------
        torch.Tensor:
            [N, 3] int32 tensor with voxel cell coordinates of points
        """
        return vg_voxel_coord_clamped(
            self.vmin, self.vmax, self.voxel_size, last_dim_ctg(points))

    def points_into_voxel_int(self, points: torch.Tensor) -> torch.Tensor:
        """
        Return point coordinates in integer voxel grid coordinates
        (from 0 to self.grid_shape).
        If point is outside of voxel grid, coordinates exceed this range.

        Parameters
        ----------
        points: torch.Tensor
            of [N, 3] shape. Points to be transformed

        Returns
        -------
        torch.Tensor:
            [N, 3] int32 tensor with voxel cell coordinates of points
        """
        return vg_voxel_coord_int(
            self.vmin, self.vmax, self.voxel_size, last_dim_ctg(points))

    def points_into_voxel_float(self, points: torch.Tensor) -> torch.Tensor:
        """
        Transform points given in world coordinates into
        voxel grid coordinates [0; self.grid_shape]
        WITHOUT rounding to nearest voxel coordinate and converting to integer.
        If point is outside of voxel grid, coordinates exceed this range.

        Parameters
        ----------
        points: torch.Tensor
            of [N, 3] shape. Points to be transformed

        Returns
        -------
        torch.Tensor:
            [N, 3] float32 tensor with voxel cell coordinates
        """
        return vg_voxel_coord_flt(self.vmin, self.voxel_size, last_dim_ctg(points))

    def voxel_coords_offset(self, points: torch.Tensor):
        """
        Transform world-space points into global index of voxel cell.
        Resulting indices are in range
        [0; self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2]]
        If point is outside of voxel grid, coordinate are clamped to the nearest
        valid voxel cell.

        Parameters
        ----------
        points: torch.Tensor
            of [N, 3] shape. Points to be transformed
        Returns
        -------
        torch.Tensor:
            [N,] uint64_t tensor with global cell indices of points
        """
        return vg_voxel_offset(
            self.vmin, self.vmax, self.voxel_size, last_dim_ctg(points))

    def save_voxel_mesh(
            self,
            tracing_result: torch.Tensor,
            path: Path | str,
            colors: torch.Tensor = None,
    ):
        """
        Save compact voxel mesh. Voxels are added to mesh if tracing_result > 0.5
        Requires "pytorch3d" and "plytorch" libraries installed.

        Parameters
        ----------
        tracing_result: torch.Tensor
            Voxel grid
        path: Path | str
            Where to save output mesh. Must have .ply extension.
        """
        try:
            from pytorch3d.ops import cubify
        except:
            raise ImportError('This operation requires "pytorch3d" to be installed')
        from plytorch import Mesh

        mesh = cubify(
            tracing_result[None].float(), 0.5,
            align='center' if colors is not None else 'corner',
            feats=colors.permute(3,0,1,2)[None]
        )

        verts = mesh.verts_list()[0]
        verts = verts[:, [2, 1, 0]] * 0.5 + 0.5
        verts = self.roi.normalized_roi_into_world(verts)
        faces = mesh.faces_list()[0]

        vert_colors = None
        if colors is not None:
            face_colors = mesh.textures.atlas_list()[0][:, 0, 0]

            vert_colors = torch.zeros_like(verts)
            norm = torch.zeros_like(verts[:, 0:1])
            vert_colors[faces[:, 0]] += face_colors
            vert_colors[faces[:, 1]] += face_colors
            vert_colors[faces[:, 2]] += face_colors
            norm[faces[:, 0]] += 1
            norm[faces[:, 1]] += 1
            norm[faces[:, 2]] += 1
            vert_colors = vert_colors / norm

        Mesh(points=verts, faces=faces, colors=vert_colors.byte()).save(path)
        return mesh, verts, faces

    def voxel_centers(self, plane='xyz', device: str = 'cpu'):
        voxel_size = self.voxel_size.to(device)
        offset = self.vmin.to(device) + voxel_size / 2
        grid_shape = self.grid_shape.tolist()

        ranges = dict(
            x=offset[0] + torch.arange(grid_shape[0], device=device) * self.voxel_size[0],
            y=offset[1] + torch.arange(grid_shape[1], device=device) * self.voxel_size[1],
            z=offset[2] + torch.arange(grid_shape[2], device=device) * self.voxel_size[2]
        )
        plane_ranges = [ranges[c] for c in plane]
        grid = torch.stack(torch.meshgrid(*plane_ranges, indexing='ij'), dim=-1)
        return grid
