import torch

from xsimgs._C import image_to_hwc_rgb_byte as image_to_hwc_rgb_byte_
from xsimgs._C import image_to_hwc_rgba_byte as image_to_hwc_rgba_byte_
from xsimgs._C import image_to_chw_rgb_byte as image_to_chw_rgb_byte_


def _check_image(image: torch.Tensor):
    assert image.ndim == 3, "Image must be [H, W, 3] tensor"
    assert image.shape[-1] == 3, "Image must be [H, W, 3] tensor"
    assert image.dtype == torch.float32, "Image must be float32 tensor"


def image_to_chw_rgb_byte(image: torch.Tensor):
    # Convert float32 [H, W, 3] image into [3, H, W] byte tensor
    _check_image(image)

    if image.is_cpu:
        return image.permute(2, 0, 1).clamp(0, 1).mul(255.0).byte()

    return image_to_chw_rgb_byte_(image.contiguous())


def image_to_hwc_rgb_byte(image: torch.Tensor):
    # Convert float32 [H, W, 3] image into [H, W, 3] byte tensor
    _check_image(image)

    if image.is_cpu:
        return image.clamp(0, 1).mul(255.0).byte()

    return image_to_hwc_rgb_byte_(image.contiguous())


def image_to_hwc_rgba_byte(image: torch.Tensor):
    # Convert float32 [H, W, 3] image into [H, W, 4] byte tensor.
    # Last channel is 255 values
    _check_image(image)

    if image.is_cpu:
        rgb = image.clamp(0, 1).mul(255.0).byte()
        alpha = torch.full(
            (rgb.shape[0], rgb.shape[1], 1),
            fill_value=255,
            dtype=torch.uint8
        )
        return torch.cat([rgb, alpha], dim=-1)

    return image_to_hwc_rgba_byte_(image.contiguous())

