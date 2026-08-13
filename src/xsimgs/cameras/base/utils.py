import numpy as np
import torch
import torch.nn.functional as F

from toast import Quat, SE3


def pixel_grid(
        width: int,
        height: int,
        device: torch.device | str = 'cpu',
):
    x = torch.arange(width, device=device, dtype=torch.int)
    x = x.view(1, -1).repeat(height, 1)
    y = torch.arange(height, device=device, dtype=torch.int)
    y = y.view(-1, 1).repeat(1, width)
    return torch.stack([x, y], dim=-1)


def uv_grid(
        width: int,
        height: int,
        with_ones: bool = False,
        device: torch.device | str = 'cpu',
        dtype: torch.dtype = torch.float32
):
    u = torch.arange(width, device=device, dtype=dtype)
    u = u.add(0.5).mul(1.0 / width).view(1, -1).repeat(height, 1)
    v = torch.arange(height, device=device, dtype=dtype)
    v = v.add(0.5).mul(1.0 / height).view(-1, 1).repeat(1, width)
    lst = [u, v, torch.ones_like(v)] if with_ones else [u, v]
    return torch.stack(lst, dim=-1)


def uv_grid_like(
        image: torch.Tensor,
        with_ones: bool = False,
        device: torch.device | str = None,
        dtype: torch.dtype = None
):
    return uv_grid(
        width=image.shape[1],
        height=image.shape[0],
        device=device if device is not None else image.device,
        dtype=dtype if dtype is not None else image.dtype,
        with_ones=with_ones
    )


def rasterize_into_depth(
        xs: torch.Tensor,
        ys: torch.Tensor,
        depth: torch.Tensor,
        width: int,
        height: int
):
    pixel_idx = ys * width + xs  # [N, ]
    # we store only point with minimal depth (closest to the viewer)
    result = torch.zeros(height * width, device=depth.device, dtype=depth.dtype)
    result = torch.scatter_reduce(
        result, index=pixel_idx, src=depth,
        dim=0, include_self=False, reduce='amin'
    )
    return result.view(height, width)


def rasterize_into_depth_mask(
        xs: torch.Tensor,
        ys: torch.Tensor,
        depth: torch.Tensor,
        width: int,
        height: int,
        near_plane: float = 0.0
):
    # returns depth map, and per-point visibility mask
    mask = (
            (ys >= 0) & (ys < height) &
            (xs >= 0) & (xs < width) &
            (depth > near_plane)
    )
    xs_masked, ys_masked, depth_masked = xs[mask], ys[mask], depth[mask]

    result = rasterize_into_depth(
        xs_masked, ys_masked, depth_masked, width=width, height=height)
    mask[mask.clone()] = depth_masked == result[ys_masked, xs_masked]
    return result, mask


def se3_look_at(
        eye: torch.Tensor | np.ndarray | list[float],
        target: torch.Tensor | np.ndarray | list[float] = None,
        dir: torch.Tensor | np.ndarray | list[float] = None,
        up = (0.0, 0.0, 1.0),
        up2 = (1.0, 0.0, 0.0)
):
    """"
    Constructs world_se3_camera transform for camera.
    Camera position is `eye` parameter
    Camera direction is defined by `dir` if given,
        or `normalized(target-eye)` vector
    Camera up is adapted from `up` vector

    Parameters
    ----------
    eye: torch.Tensor
        Position of camera
    target: torch.Tensor, optional
        Point of interest. Camera is directed toward this point.
        Either `target` or `dir` must be provided
    dir: torch.Tensor, optional
        Camera direction vector. Either `target` or `dir` must be provided
    up: torch.Tensor = (0.0, 0.0, 1.0)
        Camera up reference vector
    up2: torch.Tensor = (1.0, 0.0, 0.0)
        When camera direction is parallel to up vector,
        `up2` is used as reference vector

    Returns
    -------
    SE3
        Transformation world_se3_camera

    Warnings
    --------
    This function return correct result in the following cases:
        * Camera local space is defined by: +X right, +Y up, +Z camera direction
        * World space is defined by: +Z up, +X forward, +Y right

    For other world-space configurations result might be incorrect
    """
    eye = torch.as_tensor(eye).float()
    target = torch.as_tensor(target).float() if target is not None else None
    dir = torch.as_tensor(dir).float() if dir is not None else None
    up = torch.as_tensor(up).float() if up is not None else None
    up2 = torch.as_tensor(up2).float() if up2 is not None else None

    d = F.normalize(target - eye, dim=0) \
        if target is not None \
        else F.normalize(dir, dim=0)
    if torch.cross(d, up, dim=0).norm().item() < 1e-3:
        up = up2

    u = F.normalize(up, dim=0)
    z = d
    x = F.normalize(torch.cross(u, z, dim=0), dim=0)
    y = F.normalize(torch.cross(z, x, dim=0), dim=0)
    R = torch.as_tensor([
        [-x[0], -y[0], z[0]],
        [-x[1], -y[1], z[1]],
        [-x[2], -y[2], z[2]]
    ])
    return SE3(Quat.from_matrix(R), eye)


