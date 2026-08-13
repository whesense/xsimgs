import torch

from .distorted_perspective import DistortedPerspectiveCameraTorch


def fisheye_distortion(theta: torch.Tensor, distortion: torch.Tensor):
    # Kannala-Brandt fisheye distortion
    k1, k2, k3, k4 = distortion
    theta2 = theta * theta
    delta = theta2 * k4 + k3
    delta = theta2 * delta + k2
    delta = theta2 * delta + k1
    delta = theta2 * delta + 1.0
    return theta * delta


def fisheye_distortion_deriv(theta: torch.Tensor, distortion: torch.Tensor):
    k1, k2, k3, k4 = distortion
    theta2 = theta * theta
    d = theta2 * 9.0 * k4 + 7.0 * k3
    d = theta2 * d + 5.0 * k2
    d = theta2 * d + 3.0 * k1
    d = theta2 * d + 1.0
    return d



class FisheyeCameraTorch(DistortedPerspectiveCameraTorch):
    # 4 coefficients radial distortion: k1, k2, k3, k4
    distortion: torch.Tensor

    N_DISTORTION_ITERATIONS = 20

    def check_distortion(self):
        assert self.distortion.shape[-1] == 4, (
            "Distortion coefficients must be of [..., 8] shape: "
            "k1, k2, k3, k4")

    def camera_to_uv(self, points: torch.Tensor) -> torch.Tensor:
        xy, z = points[..., :2], points[..., 2]
        norm = xy.norm(dim=-1).clamp_min(1e-6)
        theta = torch.atan2(norm, z)
        delta = fisheye_distortion(theta, self.distortion) / norm
        uv = self.f * delta.unsqueeze(-1) * points[..., :2] + self.c

        # valid while theta_d(theta) is monotonically increasing; supports
        # FOV > 180 while rejecting the wrapped region of the polynomial
        distance = torch.where(
            (fisheye_distortion_deriv(theta, self.distortion) > 0) & (delta > 0),
            points.norm(dim=-1),
            torch.zeros_like(z)
        )
        return torch.cat([uv, distance.unsqueeze(-1)], dim=-1)

    def uv_to_camera(self, uv: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        uv_cam = (uv - self.c) / self.f
        delta = uv_cam.norm(dim=-1)

        theta = delta.clone()
        for i in range(self.N_DISTORTION_ITERATIONS):
            residual = fisheye_distortion(theta, self.distortion) - delta
            deriv = fisheye_distortion_deriv(theta, self.distortion)
            theta -= residual / deriv.clamp_min(0.1)
            theta.clamp_(0.0, torch.pi)
            if residual.abs().amax() < 1e-6:
                break

        mask = (theta > 0) & (residual.abs() < 1e-5)
        scale = theta.sin() / delta

        scale[~mask] = 0
        theta[~mask] = 0
        result = torch.cat([
            uv_cam * scale.unsqueeze(-1),
            theta.cos().unsqueeze(-1)
        ], dim=-1) * depth.unsqueeze(-1)
        return result
