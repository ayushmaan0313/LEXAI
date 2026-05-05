"""
Analyze, balance, and prepare dataset for LEXAI training.

Functions:
  - Scan data directory and report class distribution
  - Detect and report class imbalance
  - Generate stratified train/val/test CSV manifests
  - Optional undersampling of majority classes
  - Optional oversampling via image duplication for minority classes
  - Validate image integrity

Usage:
  python scripts/prepare_data.py --data_dir data                     # Analyze only
  python scripts/prepare_data.py --data_dir data --generate_manifest # Create CSV splits
  python scripts/prepare_data.py --data_dir data --balance           # Balance + manifest
  python scripts/prepare_data.py --data_dir data --validate          # Check corrupt images

Target structure:
  data/
    Normal/   *.jpg|*.png|...
    ALL/      *.jpg|*.png|...
    AML/      *.jpg|*.png|...
    CML/      *.jpg|*.png|...
"""

import argparse
import csv
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lexai.config import DEFAULT_CONFIG

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_NAMES = DEFAULT_CONFIG.data.class_names  # ["Normal", "ALL", "AML", "CML"]


def scan_data(data_dir: Path) -> Dict[str, List[Path]]:
    """Scan data directory and return {class_name: [image_paths]}."""
    class_images = {}

    folder_to_class = {
        "normal": "Normal", "benign": "Normal", "hem": "Normal",
        "all": "ALL", "all_blast": "ALL", "early": "ALL",
        "pre": "ALL", "pro": "ALL",
        "aml": "AML",
        "cml": "CML",
    }

    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir() or subdir.name == "raw":
            continue

        folder_lower = subdir.name.lower()
        cls = folder_to_class.get(folder_lower)
        if cls is None:
            for key, val in folder_to_class.items():
                if key in folder_lower:
                    cls = val
                    break
        if cls is None:
            continue

        images = sorted(
            f for f in subdir.rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            existing = class_images.get(cls, [])
            existing.extend(images)
            class_images[cls] = existing

    return class_images


def print_analysis(class_images: Dict[str, List[Path]]):
    """Print detailed class distribution analysis."""
    print(f"\n{'='*60}")
    print("  DATASET ANALYSIS")
    print(f"{'='*60}")

    total = sum(len(v) for v in class_images.values())
    if total == 0:
        print("  No images found!")
        return

    counts = {cls: len(imgs) for cls, imgs in class_images.items()}
    max_count = max(counts.values()) if counts else 0
    min_count = min(counts.values()) if counts else 0

    print(f"\n  Total images: {total:,}")
    print(f"\n  Class distribution:")
    for cls in CLASS_NAMES:
        cnt = counts.get(cls, 0)
        pct = cnt / total * 100 if total > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"    {cls:10s}: {cnt:>7,} ({pct:5.1f}%) {bar}")

    missing = [c for c in CLASS_NAMES if c not in counts or counts[c] == 0]
    if missing:
        print(f"\n  MISSING classes: {', '.join(missing)}")

    if min_count > 0:
        imbalance_ratio = max_count / min_count
        print(f"\n  Imbalance ratio: {imbalance_ratio:.1f}:1 (max/min)")
        if imbalance_ratio > 10:
            print("  WARNING: Severe imbalance (>10:1). Balancing strongly recommended.")
        elif imbalance_ratio > 3:
            print("  NOTICE: Moderate imbalance (>3:1). Balancing recommended.")
        else:
            print("  OK: Reasonable balance.")
    elif len(counts) > len(missing):
        print(f"\n  Cannot compute ratio — {len(missing)} class(es) missing.")


def validate_images(class_images: Dict[str, List[Path]]) -> int:
    """Check images for corruption. Returns count of bad images."""
    try:
        import cv2
    except ImportError:
        print("  cv2 not available, skipping validation.")
        return 0

    print(f"\n  Validating images...")
    bad = 0
    total = sum(len(v) for v in class_images.values())
    checked = 0

    for cls, images in class_images.items():
        for img_path in images:
            checked += 1
            if checked % 1000 == 0:
                print(f"    Checked {checked}/{total}...", end="\r")
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"    BAD: {img_path}")
                bad += 1

    print(f"  Validated {checked} images. Bad: {bad}")
    return bad


def balance_classes(
    class_images: Dict[str, List[Path]],
    strategy: str = "hybrid",
    target_per_class: int = None,
    seed: int = 42,
) -> Dict[str, List[Path]]:
    """
    Balance class distribution.

    Strategies:
      - undersample: reduce majority classes to match minority
      - oversample: duplicate minority class images to match majority
      - hybrid: undersample majority to median, oversample minority to median
    """
    rng = random.Random(seed)
    counts = {cls: len(imgs) for cls, imgs in class_images.items()}

    if not counts:
        return class_images

    if target_per_class is not None:
        target = target_per_class
    elif strategy == "undersample":
        target = min(counts.values())
    elif strategy == "oversample":
        target = max(counts.values())
    else:  # hybrid
        sorted_counts = sorted(counts.values())
        target = sorted_counts[len(sorted_counts) // 2]

    print(f"\n  Balancing: strategy={strategy}, target={target:,} per class")

    balanced = {}
    for cls in CLASS_NAMES:
        images = class_images.get(cls, [])
        if not images:
            print(f"    {cls}: SKIPPED (no images)")
            continue

        current = len(images)
        if current > target:
            sampled = rng.sample(images, target)
            balanced[cls] = sampled
            print(f"    {cls}: {current:,} -> {target:,} (undersampled)")
        elif current < target:
            repeats = target // current
            remainder = target % current
            oversampled = images * repeats + rng.sample(images, remainder)
            balanced[cls] = oversampled
            print(f"    {cls}: {current:,} -> {target:,} (oversampled {repeats}x)")
        else:
            balanced[cls] = list(images)
            print(f"    {cls}: {current:,} (unchanged)")

    return balanced


def generate_manifest(
    class_images: Dict[str, List[Path]],
    output_dir: Path,
    train_split: float = 0.7,
    val_split: float = 0.15,
    seed: int = 42,
):
    """Generate stratified train/val/test CSV manifests."""
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_map = {name: i for i, name in enumerate(CLASS_NAMES)}

    splits = {"train": [], "val": [], "test": []}

    for cls, images in sorted(class_images.items()):
        label = label_map.get(cls)
        if label is None:
            continue

        shuffled = list(images)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * train_split)
        n_val = int(n * val_split)

        for img in shuffled[:n_train]:
            splits["train"].append((str(img), label))
        for img in shuffled[n_train:n_train + n_val]:
            splits["val"].append((str(img), label))
        for img in shuffled[n_train + n_val:]:
            splits["test"].append((str(img), label))

    print(f"\n  Generating manifests:")
    for split_name, entries in splits.items():
        rng.shuffle(entries)
        csv_path = output_dir / f"{split_name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            for path, label in entries:
                writer.writerow([path, label])

        label_counts = Counter(label for _, label in entries)
        dist_str = ", ".join(
            f"{CLASS_NAMES[i]}={label_counts.get(i, 0)}"
            for i in range(len(CLASS_NAMES))
        )
        print(f"    {split_name:5s}: {len(entries):>6,} images ({dist_str})")
        print(f"           -> {csv_path}")

    print(f"\n  Train with manifests:")
    print(f"    python scripts/train.py --data_dir {output_dir.parent} "
          f"--manifest {output_dir / 'train.csv'}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze, balance, and prepare LEXAI dataset"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to data directory with class subdirectories",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Check images for corruption",
    )
    parser.add_argument(
        "--balance", action="store_true",
        help="Balance classes before generating manifest",
    )
    parser.add_argument(
        "--strategy", type=str, default="hybrid",
        choices=["undersample", "oversample", "hybrid"],
        help="Balancing strategy (default: hybrid)",
    )
    parser.add_argument(
        "--target_per_class", type=int, default=None,
        help="Override: exact number of images per class",
    )
    parser.add_argument(
        "--generate_manifest", action="store_true",
        help="Generate train/val/test CSV manifests",
    )
    parser.add_argument(
        "--manifest_dir", type=str, default=None,
        help="Directory for manifest CSVs (default: data_dir/manifests)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist.")
        print(f"Download data first: python scripts/download_dataset.py --all")
        sys.exit(1)

    class_images = scan_data(data_dir)
    print_analysis(class_images)

    if args.validate:
        validate_images(class_images)

    if args.balance:
        class_images = balance_classes(
            class_images,
            strategy=args.strategy,
            target_per_class=args.target_per_class,
            seed=args.seed,
        )
        args.generate_manifest = True

    if args.generate_manifest or args.balance:
        manifest_dir = Path(args.manifest_dir) if args.manifest_dir else data_dir / "manifests"
        generate_manifest(class_images, manifest_dir, seed=args.seed)
    else:
        print(f"\n  To generate manifests: add --generate_manifest")
        print(f"  To balance + generate: add --balance")


if __name__ == "__main__":
    main()
