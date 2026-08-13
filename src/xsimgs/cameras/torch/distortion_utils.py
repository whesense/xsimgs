import torch


def uv_distortion(uv: torch.Tensor, dist: torch.Tensor):
    k1, k2, p1, p2, k5, k6, k7, k8 = dist

    uu = uv[..., 0] * uv[..., 0]
    vv = uv[..., 1] * uv[..., 1]
    r2 = uu + vv
    r4 = r2 * r2
    r6 = r4 * r2

    rd = 1.0 + r2 * k1 + r4 * k2 + r6 * k5
    rd_denom = 1.0 + r2 * k6 + r4 * k7 + r6 * k8
    radial_scale = (rd / rd_denom).unsqueeze(-1)

    u_mul_v = uv[..., 0] * uv[..., 1]

    u_tangential = 2.0 * p1 * u_mul_v + p2 * (r2 + 2.0 * uu)
    v_tangential = 2.0 * p2 * u_mul_v + p1 * (r2 + 2.0 * vv)
    tang_offset = torch.stack([u_tangential, v_tangential], dim=-1)
    return radial_scale, tang_offset


# Below this the forward distortion is close to folding and a newton step would
# run away; a fixed-point step is taken there instead. See the CUDA twin in
# csrc/cameras/distorted_perspective.cu, which this mirrors step for step.
MIN_JACOBIAN_DET = 1e-3


def iterate_undistortion(
        uv: torch.Tensor,
        dist: torch.Tensor,
        num_iterations: int = 5,
        min_jacobian_det: float = MIN_JACOBIAN_DET
):
    """Invert `uv_distortion`, solving it by Newton-Raphson on the residual.

    Finds the `uv` whose distortion lands on the given (distorted) `uv`, i.e.
    drives `uv * radial(uv) + tangential(uv) - uv_distorted` to zero, stepping
    by that residual times the inverse of the analytic jacobian below. Newton
    converges quadratically, so five steps reach float32 precision even for the
    rational (k4..k6) models; a fixed-point iteration on the same residual
    converges linearly and needs tens of steps for those.

    The residual is always taken against the distorted point being inverted.
    Substituting the running estimate for it turns the solve into a search for
    `radial(uv) == 1`, whose fixed point is a single radius (`r^2 = -k1/k2` for
    a two-term model) that every input collapses onto regardless of where it
    started.
    """
    if dist.abs().sum() == 0:
        return uv

    k1, k2, p1, p2, k3, k4, k5, k6 = dist

    uv_distorted = uv
    for i in range(num_iterations):
        u, v = uv[..., 0], uv[..., 1]
        r2 = u * u + v * v

        num = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        den = 1.0 + r2 * (k4 + r2 * (k5 + r2 * k6))
        scale = num / den

        # d(scale)/d(r2), by the quotient rule
        d_num = k1 + r2 * (2.0 * k2 + 3.0 * k3 * r2)
        d_den = k4 + r2 * (2.0 * k5 + 3.0 * k6 * r2)
        d_scale = (d_num * den - num * d_den) / (den * den)

        delta = torch.stack([
            2.0 * p1 * u * v + p2 * (r2 + 2.0 * u * u),
            p1 * (r2 + 2.0 * v * v) + 2.0 * p2 * u * v
        ], dim=-1)
        residual = uv * scale.unsqueeze(-1) + delta - uv_distorted

        # Jacobian of uv * scale(uv) + delta(uv); its off-diagonals coincide.
        j_uu = scale + 2.0 * u * u * d_scale + 2.0 * p1 * v + 6.0 * p2 * u
        j_uv = 2.0 * u * v * d_scale + 2.0 * p1 * u + 2.0 * p2 * v
        j_vv = scale + 2.0 * v * v * d_scale + 6.0 * p1 * v + 2.0 * p2 * u
        det = j_uu * j_vv - j_uv * j_uv

        res_u, res_v = residual[..., 0], residual[..., 1]
        newton = uv - torch.stack([
            (j_vv * res_u - j_uv * res_v) / det,
            (j_uu * res_v - j_uv * res_u) / det
        ], dim=-1)
        fixed_point = (uv_distorted - delta) / scale.unsqueeze(-1)

        uv = torch.where(
            (det.abs() > min_jacobian_det).unsqueeze(-1), newton, fixed_point
        )

    return uv


def max_valid_radius(
        normalized_intrinsics: torch.Tensor,
        distortion: torch.Tensor,
        margin: float = 1.5,
        search_limit: float = 16.0,
        num_samples: int = 8192
) -> torch.Tensor:
    """Largest camera-space radius each calibration may be evaluated at.

    Radius here is `r = |xy| / z`, the tangent of the angle off the optical
    axis. Returns one value per camera, shaped like `distortion` with its last
    dimension replaced by 1.

    Two limits, whichever binds first:

    * **The fold.** The radial map `rho(r) = r * scale(r)` must strictly
      increase -- a ray further off axis has to land further from the principal
      point, or the lens is not a lens. The coefficients are fitted inside the
      image and merely extrapolate outside it, so `rho` may turn over at some
      radius. Past that point two ray angles share an image point: geometry far
      outside the field of view projects onto a plausible pixel, at a plausible
      depth, and renders as a ghost that nothing downstream can identify. Past
      it further `rho` crosses zero and mirrors the far field through the
      principal point.
    * **`margin` times the image corner.** Beyond the corner the model
      extrapolates from data no calibration ever saw. The margin leaves room for
      gaussians centred outside the frame that still splat into it.

    Note the fold is a property of the coefficients alone, while the corner
    follows from the intrinsics; a long lens can reach its corner well before
    any fold, and a wide one can fold just past its corner.
    """
    dist = distortion.reshape(-1, 8)
    intrinsics = normalized_intrinsics.reshape(-1, 4)
    k1, k2, p1, p2, k3, k4, k5, k6 = dist.unbind(-1)

    r = torch.linspace(
        0.0, search_limit, num_samples, device=dist.device, dtype=dist.dtype)
    r2 = (r * r).unsqueeze(0)
    num = 1.0 + r2 * (k1[:, None] + r2 * (k2[:, None] + r2 * k3[:, None]))
    den = 1.0 + r2 * (k4[:, None] + r2 * (k5[:, None] + r2 * k6[:, None]))
    rho = r.unsqueeze(0) * (num / den)

    # First sample where rho stops increasing; where it never does, the whole
    # search range is admissible.
    decreasing = rho[:, 1:] < rho[:, :-1]
    fold_idx = torch.where(
        decreasing.any(dim=-1),
        decreasing.to(torch.uint8).argmax(dim=-1),
        torch.full_like(decreasing[:, 0], num_samples - 1, dtype=torch.long)
    )
    radius_fold = r[fold_idx]

    # The image corner, carried back through the distortion. Normalized
    # intrinsics already account for the aspect ratio, so this needs no
    # resolution. Searching the running maximum of rho keeps the lookup on a
    # sorted sequence even for calibrations that fold inside the search range.
    f, c = intrinsics[:, :2], intrinsics[:, 2:]
    corners = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        device=dist.device, dtype=dist.dtype)
    rho_corner = (((corners.unsqueeze(0) - c.unsqueeze(1)) / f.unsqueeze(1))
                  .norm(dim=-1).amax(dim=-1))
    corner_idx = torch.searchsorted(
        rho.cummax(dim=-1).values.contiguous(), rho_corner.unsqueeze(-1)
    ).squeeze(-1).clamp(max=num_samples - 1)
    radius_corner = r[corner_idx]

    radius = torch.minimum(radius_fold, margin * radius_corner)
    return radius.reshape(*distortion.shape[:-1], 1)
