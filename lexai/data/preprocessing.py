"""Image preprocessing: stain normalization, transforms, utility functions."""

from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from lexai.config import DataConfig, DEFAULT_CONFIG


class MacenkoNormalizer:
    """
    Macenko SVD-based H&E stain normalization.
    Reduces inter-lab color variation in histopathology images.
    """

    def __init__(self, Io: float = 240, alpha: float = 1.0, beta: float = 0.15):
        self.Io = Io
        self.alpha = alpha
        self.beta = beta
        self.HERef = np.array([
            [0.5626, 0.2159],
            [0.7201, 0.8012],
            [0.4062, 0.5581],
        ])
        self.maxCRef = np.array([1.9705, 1.0308])

    def _get_stain_matrix(self, img: np.ndarray):
        img_float = img.reshape((-1, 3)).astype(np.float64)
        img_float = np.maximum(img_float, 1)
        OD = -np.log(img_float / self.Io)

        ODhat = OD[~np.any(OD < self.beta, axis=1)]
        if len(ODhat) == 0:
            return self.HERef, self.maxCRef, None

        _, _, V = np.linalg.svd(ODhat, full_matrices=False)
        V = V[:2].T

        That = ODhat @ V
        phi = np.arctan2(That[:, 1], That[:, 0])
        minPhi = np.percentile(phi, self.alpha)
        maxPhi = np.percentile(phi, 100 - self.alpha)

        vMin = V @ np.array([np.cos(minPhi), np.sin(minPhi)])
        vMax = V @ np.array([np.cos(maxPhi), np.sin(maxPhi)])

        HE = np.array([vMin * np.sign(vMin[0]), vMax * np.sign(vMax[0])]).T

        Y = OD.T
        C = np.linalg.lstsq(HE, Y, rcond=None)[0]
        maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])

        return HE, maxC, OD

    def fit(self, img: np.ndarray):
        HE, maxC, _ = self._get_stain_matrix(img)
        self.HERef = HE
        self.maxCRef = maxC

    def transform(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        HE, maxC, OD = self._get_stain_matrix(img)
        if OD is None:
            return img

        Y = OD.T
        C = np.linalg.lstsq(HE, Y, rcond=None)[0]
        C = C / maxC[:, np.newaxis] * self.maxCRef[:, np.newaxis]
        OD_normalised = self.HERef @ C
        with np.errstate(over="ignore"):
            img_normalised = self.Io * np.exp(-OD_normalised.T)
        img_normalised = np.clip(img_normalised, 0, 255).astype(np.uint8)
        return img_normalised.reshape(h, w, 3)


def stain_normalize_reinhard(image: np.ndarray) -> np.ndarray:
    """Simple Reinhard stain normalization in LAB color space."""
    import cv2

    target_mean = np.array([148.60, 169.30, 105.97])
    target_std = np.array([41.56, 9.01, 6.67])

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float64)
    for i in range(3):
        src_mean = lab[:, :, i].mean()
        src_std = lab[:, :, i].std() + 1e-6
        lab[:, :, i] = (
            (lab[:, :, i] - src_mean) / src_std * target_std[i] + target_mean[i]
        )

    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def get_inference_transform(config: DataConfig = None) -> transforms.Compose:
    config = config or DEFAULT_CONFIG.data
    return transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std),
    ])


def get_train_transform(config: DataConfig = None) -> transforms.Compose:
    config = config or DEFAULT_CONFIG.data
    size = config.image_size
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), shear=5),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        transforms.Normalize(mean=config.mean, std=config.std),
    ])


def denormalize(
    tensor: torch.Tensor,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    t = tensor.clone()
    for c in range(3):
        t[c] = t[c] * std[c] + mean[c]
    return t.clamp(0, 1)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    denorm = denormalize(tensor)
    arr = denorm.permute(1, 2, 0).cpu().numpy()
    return (arr * 255).astype(np.uint8)


def preprocess_image(
    image: Image.Image,
    config: DataConfig = None,
) -> torch.Tensor:
    transform = get_inference_transform(config)
    return transform(image).unsqueeze(0)
