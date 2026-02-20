"""Dataset loader for leukemia blood smear images."""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms

from lexai.config import DataConfig, DEFAULT_CONFIG


class LeukemiaDataset(Dataset):
    """
    Dataset for leukemia blood smear images.

    Expects directory structure:
        data_dir/
            ALL_Blast/
            ALL_Early_Pre_B/
            ALL_Pre_B/
            ALL_Pro_B/
            Benign/
    """

    def __init__(
        self,
        data_dir: str,
        config: DataConfig = None,
        transform: Optional[transforms.Compose] = None,
        split: str = "train",
    ):
        self.data_dir = Path(data_dir)
        self.config = config or DEFAULT_CONFIG.data
        self.split = split
        self.samples: List[Tuple[str, int]] = []  # (image_path, label)
        self.class_to_idx: Dict[str, int] = {}

        # Build class mapping
        for idx, class_name in enumerate(self.config.class_names):
            self.class_to_idx[class_name] = idx

        # Scan directories
        self._load_samples()

        # Set up transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._default_transform()

    def _load_samples(self):
        """Scan class directories and collect image paths."""
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue

            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in valid_extensions:
                    self.samples.append((str(img_path), class_idx))

    def _default_transform(self) -> transforms.Compose:
        """Build augmentation pipeline — aggressive for training to prevent overfitting."""
        size = self.config.image_size

        if self.split == "train" and self.config.augmentation:
            return transforms.Compose([
                # RandomResizedCrop forces model to learn from partial views
                transforms.RandomResizedCrop(
                    size, scale=(0.6, 1.0), ratio=(0.8, 1.2)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=30),
                # Stronger color jitter to generalize across staining
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4,
                    saturation=0.4, hue=0.1
                ),
                transforms.RandomAffine(
                    degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15),
                    shear=10,
                ),
                # Gaussian blur for robustness to focus
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                # Random erasing forces model to use global features
                transforms.RandomErasing(
                    p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)
                ),
                transforms.Normalize(
                    mean=self.config.mean, std=self.config.std
                ),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config.mean, std=self.config.std
                ),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Returns:
            image: Tensor of shape (3, H, W)
            label: int class label
            path: original image path (for explainability tracing)
        """
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label, img_path


def compute_class_weights(
    samples: List[Tuple[str, int]],
    num_classes: int,
) -> torch.Tensor:
    """Compute inverse-frequency class weights for balanced training."""
    counts = torch.zeros(num_classes)
    for _, label in samples:
        counts[label] += 1

    # Inverse frequency, normalized to sum=num_classes
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return weights


def create_data_loaders(
    data_dir: str,
    config: DataConfig = None,
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 42,
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test data loaders with stratified splitting
    and class-balanced sampling.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    config = config or DEFAULT_CONFIG.data

    # Load full dataset (no augmentation for splitting)
    full_dataset = LeukemiaDataset(data_dir, config, split="train")

    if len(full_dataset) == 0:
        raise ValueError(
            f"No images found in {data_dir}. "
            f"Expected subdirectories: {config.class_names}"
        )

    # Stratified split
    indices_per_class: Dict[int, List[int]] = {}
    for idx, (_, label) in enumerate(full_dataset.samples):
        indices_per_class.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    train_indices, val_indices, test_indices = [], [], []

    print("\n  Class distribution:")
    for cls_idx, indices in sorted(indices_per_class.items()):
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(n * config.train_split)
        n_val = int(n * config.val_split)

        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])

        cls_name = config.class_names[cls_idx] if cls_idx < len(config.class_names) else f"class_{cls_idx}"
        print(f"    {cls_name:20s}: {n:>6} total | "
              f"{n_train:>5} train | {n_val:>4} val | {n - n_train - n_val:>4} test")

    # Create split datasets with appropriate transforms
    train_ds = LeukemiaDataset(data_dir, config, split="train")
    val_ds = LeukemiaDataset(data_dir, config, split="val")
    test_ds = LeukemiaDataset(data_dir, config, split="test")

    # Weighted sampler for balanced training batches
    sampler = None
    shuffle = True
    if use_weighted_sampler:
        # Compute per-sample weights based on class frequency
        class_counts = {}
        for i in train_indices:
            _, label = full_dataset.samples[i]
            class_counts[label] = class_counts.get(label, 0) + 1

        sample_weights = []
        for i in train_indices:
            _, label = full_dataset.samples[i]
            # Weight = 1 / class_count (so rare classes are sampled more)
            sample_weights.append(1.0 / class_counts[label])

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_indices),
            replacement=True,
        )
        shuffle = False  # Sampler and shuffle are mutually exclusive
        print(f"\n  Using WeightedRandomSampler for class balance")
        for cls_idx in sorted(class_counts.keys()):
            cls_name = config.class_names[cls_idx]
            w = 1.0 / class_counts[cls_idx]
            print(f"    {cls_name:20s}: count={class_counts[cls_idx]:>5}  weight={w:.6f}")

    train_loader = DataLoader(
        Subset(train_ds, train_indices),
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(test_ds, test_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
