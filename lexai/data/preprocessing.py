"""Image preprocessing utilities for blood smear analysis."""

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from lexai.config import DataConfig, DEFAULT_CONFIG


def get_inference_transform(config: DataConfig = None) -> transforms.Compose:
    """Get the transform pipeline for inference (no augmentation)."""
    config = config or DEFAULT_CONFIG.data
    return transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std),
    ])


def denormalize(
    tensor: torch.Tensor,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Reverse ImageNet normalization for visualization."""
    t = tensor.clone()
    for c in range(3):
        t[c] = t[c] * std[c] + mean[c]
    return t.clamp(0, 1)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a (C, H, W) tensor to an (H, W, C) uint8 numpy array."""
    denorm = denormalize(tensor)
    arr = denorm.permute(1, 2, 0).cpu().numpy()
    return (arr * 255).astype(np.uint8)


def preprocess_image(
    image: Image.Image,
    config: DataConfig = None,
) -> torch.Tensor:
    """
    Preprocess a PIL Image for model inference.

    Returns:
        Tensor of shape (1, 3, H, W)
    """
    transform = get_inference_transform(config)
    return transform(image).unsqueeze(0)


def stain_normalize(image: np.ndarray) -> np.ndarray:
    """
    Simple stain normalization using Reinhard method.
    Normalizes to a target mean/std in LAB color space.
    
    Args:
        image: (H, W, 3) uint8 BGR image
    
    Returns:
        Normalized (H, W, 3) uint8 BGR image
    """
    import cv2

    # Target statistics (typical H&E stain)
    target_mean = np.array([148.60, 169.30, 105.97])
    target_std = np.array([41.56, 9.01, 6.67])

    # Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float64)

    # Compute source stats per channel
    for i in range(3):
        src_mean = lab[:, :, i].mean()
        src_std = lab[:, :, i].std() + 1e-6
        lab[:, :, i] = (
            (lab[:, :, i] - src_mean) / src_std * target_std[i]
            + target_mean[i]
        )

    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
