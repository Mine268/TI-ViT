from einops import rearrange
import cv2
import numpy as np
import torch


def denormalize(
    img: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    clamp_output: bool = False
) -> torch.Tensor:
    """Enhanced image denormalization with device check and safety features

    Args:
        img: Input tensor of shape [B, C, H, W] or [C, H, W]
        mean: Channel means of shape [C]
        std: Channel stds of shape [C]
        clamp_output: Whether to clamp to [0,1] range

    Returns:
        Denormalized tensor with same shape as input
    """
    # Device consistency check
    if img.device != mean.device or img.device != std.device:
        raise RuntimeError(
            f"Device mismatch: img({img.device}), mean({mean.device}), std({std.device})")

    # Dimension expansion for broadcasting
    dims = (3, 1, 1) if img.ndim == 3 else (1, 3, 1, 1)
    mean = mean.view(*dims)
    std = std.view(*dims)

    # Numerical stability
    safe_std = std.clone()
    safe_std[safe_std < 1e-7] = 1.0  # Prevent division by zero

    # Core computation
    denorm_img = img * safe_std + mean

    # Optional value clamping
    if clamp_output:
        denorm_img = torch.clamp(denorm_img, 0.0, 1.0)

    return denorm_img


def save_tensor_img(img_tensor: torch.Tensor, img_path: str):
    if img_tensor.ndim != 3:
        raise ValueError("Imcompatible tensor size for image: " + img_tensor.shape)

    cv2.imwrite(img_path,
        rearrange(img_tensor.detach().cpu().numpy() * 255, "c h w -> h w c").astype(np.uint8))