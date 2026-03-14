"""
Reorganize the LEXAI data directory from ALL-subtypes to a unified class structure.

Converts:
  data/ALL_Blast/         ─┐
  data/ALL_Early_Pre_B/    ├──→  data/ALL/     (merged)
  data/ALL_Pre_B/          │
  data/ALL_Pro_B/         ─┘
  data/Benign/            ────→  data/Normal/  (renamed)
  data/AML/               ────→  data/AML/     (unchanged)

Usage:
  py scripts/reorganize_for_aml.py --data_dir data              # Execute
  py scripts/reorganize_for_aml.py --data_dir data --dry_run    # Preview only
"""

import argparse
import shutil
import sys
from pathlib import Path
from collections import defaultdict

# ALL subtype directories to merge
ALL_SUBTYPES = ["ALL_Blast", "ALL_Early_Pre_B", "ALL_Pre_B", "ALL_Pro_B"]

# Directories to rename
RENAME_MAP = {"Benign": "Normal"}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def count_images(directory: Path) -> int:
    """Count image files in a directory."""
    if not directory.exists():
        return 0
    return sum(
        1 for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def print_summary(data_dir: Path, label: str = "Current"):
    """Print class distribution summary."""
    print(f"\n  {label} Layout:")
    total = 0
    for d in sorted(data_dir.iterdir()):
        if d.is_dir() and d.name != "raw":
            count = count_images(d)
            if count > 0:
                print(f"    {d.name:25s}: {count:>6,} images")
                total += count
    print(f"    {'─' * 40}")
    print(f"    {'TOTAL':25s}: {total:>6,} images")
    return total


def reorganize(data_dir: Path, dry_run: bool = False):
    """Reorganize data directory from ALL subtypes to unified classes."""
    data_dir = Path(data_dir)

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    print("=" * 55)
    print("  LEXAI Data Reorganization")
    print("=" * 55)

    print_summary(data_dir, "Before")

    if dry_run:
        print("\n  🔍 DRY RUN — no files will be moved\n")

    # --- Merge ALL subtypes ---
    all_dir = data_dir / "ALL"
    all_count = 0

    subtype_dirs = [data_dir / name for name in ALL_SUBTYPES if (data_dir / name).exists()]

    if subtype_dirs:
        if not dry_run:
            all_dir.mkdir(parents=True, exist_ok=True)

        for subtype_dir in subtype_dirs:
            images = [
                f for f in subtype_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            ]

            for img in images:
                # Prefix with subtype to avoid name collisions
                prefix = subtype_dir.name.lower().replace(" ", "_")
                dest = all_dir / f"{prefix}_{img.name}"

                if dry_run:
                    pass  # Just count
                else:
                    shutil.copy2(str(img), str(dest))

                all_count += 1

            if not dry_run:
                shutil.rmtree(str(subtype_dir))
                print(f"  ✓ Merged {subtype_dir.name}/ → ALL/ ({len(images)} images)")
            else:
                print(f"  [DRY] Would merge {subtype_dir.name}/ → ALL/ ({len(images)} images)")

        print(f"\n  Total ALL images: {all_count:,}")
    else:
        existing_all = count_images(all_dir)
        if existing_all > 0:
            print(f"\n  ALL/ already exists with {existing_all:,} images (no merge needed)")
        else:
            print("\n  ⚠ No ALL subtype directories found to merge")

    # --- Rename Benign → Normal ---
    for old_name, new_name in RENAME_MAP.items():
        old_dir = data_dir / old_name
        new_dir = data_dir / new_name

        if old_dir.exists():
            old_count = count_images(old_dir)

            if new_dir.exists():
                # Merge into existing Normal/
                images = [
                    f for f in old_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
                ]
                for img in images:
                    dest = new_dir / f"benign_{img.name}"
                    if not dry_run:
                        shutil.copy2(str(img), str(dest))

                if not dry_run:
                    shutil.rmtree(str(old_dir))
                    print(f"\n  ✓ Merged {old_name}/ into existing Normal/ ({old_count} images)")
                else:
                    print(f"\n  [DRY] Would merge {old_name}/ into existing Normal/ ({old_count} images)")
            else:
                if not dry_run:
                    old_dir.rename(new_dir)
                    print(f"\n  ✓ Renamed {old_name}/ → {new_name}/ ({old_count} images)")
                else:
                    print(f"\n  [DRY] Would rename {old_name}/ → {new_name}/ ({old_count} images)")

    # --- Check AML ---
    aml_dir = data_dir / "AML"
    aml_count = count_images(aml_dir)
    if aml_count > 0:
        print(f"\n  ✓ AML/ already contains {aml_count:,} images")
    else:
        print(f"\n  ⚠ AML/ directory is empty or missing.")
        print(f"    Download the AML dataset first:")
        print(f"      py scripts/download_tcia.py --output_dir {data_dir}")

    # --- Final summary ---
    if not dry_run:
        print_summary(data_dir, "After")
        print(f"\n✅ Reorganization complete!")
        print(f"   Train with: py scripts/train.py --data_dir {data_dir} --preset aml --epochs 50")
    else:
        print(f"\n  Run without --dry_run to apply changes.")


def main():
    parser = argparse.ArgumentParser(
        description="Reorganize LEXAI data from ALL subtypes to unified class structure"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to the data directory"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Preview changes without modifying any files"
    )
    args = parser.parse_args()
    reorganize(args.data_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
