"""Unified dataset loader for leukemia blood smear images."""

import csv
import os
import platform
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms

from lexai.config import DataConfig, DEFAULT_CONFIG
from lexai.data.preprocessing import (
    MacenkoNormalizer,
    get_train_transform,
    get_inference_transform,
)

_DEFAULT_NUM_WORKERS = 0 if platform.system() == "Windows" else 4

CLASS_NAMES = ["Normal", "ALL", "AML", "CML"]

_FOLDER_TO_LABEL = {
    "normal": 0, "benign": 0, "hem": 0,
    "all": 1, "all_blast": 1, "early": 1, "pre": 1, "pro": 1,
    "all_early_pre_b": 1, "all_pre_b": 1, "all_pro_b": 1,
    "aml": 2,
    "cml": 3,
}


class LeukemiaDataset(Dataset):
    """
    Dataset for leukemia blood smear images.

    Supports two modes:
    - Directory scanning: data_dir/{Normal,ALL,AML,CML}/**/*.{jpg,png,...}
    - CSV manifest: path,label per line
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform=None,
        normalizer: Optional[MacenkoNormalizer] = None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))

        if image is None:
            warnings.warn(f"Could not read image: {img_path} — using blank")
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.normalizer is not None:
            try:
                image = self.normalizer.transform(image)
            except Exception:
                pass

        pil_image = Image.fromarray(image)
        if self.transform:
            image = self.transform(pil_image)
        else:
            image = transforms.ToTensor()(pil_image)

        return image, self.labels[idx], img_path


def scan_directory(
    data_dir: str,
    config: DataConfig = None,
) -> Tuple[List[str], List[int]]:
    """
    Scan a directory tree for images, mapping folder names to class labels.

    Handles various folder naming conventions (case-insensitive):
    - Normal/Benign/HEM -> 0
    - ALL/ALL_Blast/Early/Pre/Pro -> 1
    - AML -> 2
    - CML -> 3
    """
    config = config or DEFAULT_CONFIG.data
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_paths = []
    labels = []

    data_path = Path(data_dir)
    if not data_path.exists():
        return image_paths, labels

    for subdir in sorted(data_path.iterdir()):
        if not subdir.is_dir():
            continue

        folder_name = subdir.name.lower()
        label = _FOLDER_TO_LABEL.get(folder_name)

        if label is None:
            for key, val in _FOLDER_TO_LABEL.items():
                if key in folder_name:
                    label = val
                    break

        if label is None:
            continue

        for img_path in sorted(subdir.rglob("*")):
            if img_path.suffix.lower() in valid_extensions:
                image_paths.append(str(img_path))
                labels.append(label)

    return image_paths, labels


def load_from_manifest(csv_path: str) -> Tuple[List[str], List[int]]:
    """Load dataset from a CSV manifest (path,label per line)."""
    image_paths = []
    labels = []
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                image_paths.append(row[0])
                labels.append(int(row[1]))
    return image_paths, labels


def save_manifest(image_paths: List[str], labels: List[int], path: str):
    """Save a CSV manifest for reproducibility."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for p, l in zip(image_paths, labels):
            writer.writerow([p, l])


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes)
    for label in labels:
        counts[label] += 1
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return weights


def fit_stain_normalizer(image_paths: List[str], n_ref: int = 50) -> MacenkoNormalizer:
    """Fit a Macenko normalizer on a sample of training images."""
    normalizer = MacenkoNormalizer()
    ref_images = []
    ref_size = (256, 256)

    for p in image_paths[:n_ref]:
        img = cv2.imread(p)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, ref_size)
            ref_images.append(img_rgb)

    if ref_images:
        ref = np.median(np.stack(ref_images), axis=0).astype(np.uint8)
        normalizer.fit(ref)
        print("  MacenkoNormalizer fitted on training set")

    return normalizer


def create_data_loaders(
    data_dir: str,
    config: DataConfig = None,
    batch_size: int = 16,
    num_workers: int = _DEFAULT_NUM_WORKERS,
    seed: int = 42,
    manifest_csv: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test data loaders with stratified splitting,
    optional stain normalization, and class-balanced sampling.
    """
    config = config or DEFAULT_CONFIG.data

    if manifest_csv and os.path.isfile(manifest_csv):
        image_paths, labels = load_from_manifest(manifest_csv)
    else:
        image_paths, labels = scan_directory(data_dir, config)

    if not image_paths:
        raise ValueError(
            f"No images found in {data_dir}. "
            f"Expected subdirectories matching: {list(_FOLDER_TO_LABEL.keys())}"
        )

    # Print class distribution
    class_counts = {}
    for l in labels:
        name = config.class_names[l] if l < len(config.class_names) else f"class_{l}"
        class_counts[name] = class_counts.get(name, 0) + 1
    print(f"\n  Dataset: {len(image_paths)} images")
    for name, count in sorted(class_counts.items()):
        print(f"    {name:12s}: {count:>6}")

    # Stratified split
    indices_per_class: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        indices_per_class.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    train_indices, val_indices, test_indices = [], [], []

    for cls_idx, indices in sorted(indices_per_class.items()):
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(n * config.train_split)
        n_val = int(n * config.val_split)

        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train : n_train + n_val])
        test_indices.extend(indices[n_train + n_val :])

    # Split paths/labels
    train_paths = [image_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_paths = [image_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    test_paths = [image_paths[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]

    print(f"  Split: {len(train_paths)} train / {len(val_paths)} val / {len(test_paths)} test")

    # Stain normalization
    normalizer = None
    if config.use_stain_norm:
        normalizer = fit_stain_normalizer(train_paths)

    # Transforms
    train_transform = get_train_transform(config)
    val_transform = get_inference_transform(config)

    train_ds = LeukemiaDataset(train_paths, train_labels, train_transform, normalizer)
    val_ds = LeukemiaDataset(val_paths, val_labels, val_transform, normalizer)
    test_ds = LeukemiaDataset(test_paths, test_labels, val_transform, normalizer)

    # Weighted sampler
    sampler = None
    shuffle = True
    if config.num_classes > 1:
        class_counts_train = np.bincount(train_labels, minlength=config.num_classes)
        sample_weights = [1.0 / (class_counts_train[l] + 1e-6) for l in train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_labels),
            replacement=True,
        )
        shuffle = False
        print("  Using WeightedRandomSampler for class balance")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
