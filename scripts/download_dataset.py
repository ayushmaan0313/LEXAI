"""
Download and prepare large leukemia datasets from Kaggle.

Supported datasets (combined ~48K+ images):
  1. C-NMC 2019: 15,135 images (ALL blast vs Normal)
  2. AML Cytomorphology: 18,365 images (AML subtypes)
  3. Blood Cell Cancer Detection: 5,000 images (ALL, AML, Normal + subtypes)
  4. ALL Image Dataset: 3,256 images (ALL subtypes vs benign)

Usage:
  py -3.10 scripts/download_dataset.py --all
  py -3.10 scripts/download_dataset.py --dataset cnmc
  py -3.10 scripts/download_dataset.py --dataset aml
  py -3.10 scripts/download_dataset.py --dataset blood_cell
  py -3.10 scripts/download_dataset.py --dataset all_subtypes

Requires:
  - Kaggle API credentials at ~/.kaggle/kaggle.json
  - pip install kaggle
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_kaggle_credentials():
    """Check if Kaggle API credentials are configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("=" * 60)
        print("KAGGLE API CREDENTIALS NOT FOUND")
        print("=" * 60)
        print()
        print("To download datasets, you need Kaggle API credentials:")
        print()
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Click 'Create New Token' under the API section")
        print("3. This downloads a kaggle.json file")
        print(f"4. Place it at: {kaggle_json}")
        print()
        print("The file should contain:")
        print('  {"username":"your_username","key":"your_api_key"}')
        print()

        # Offer manual setup
        username = input("Or enter your Kaggle username now (or press Enter to skip): ").strip()
        if username:
            key = input("Enter your Kaggle API key: ").strip()
            if key:
                kaggle_json.parent.mkdir(parents=True, exist_ok=True)
                kaggle_json.write_text(f'{{"username":"{username}","key":"{key}"}}')
                os.chmod(str(kaggle_json), 0o600)
                print(f"\nCredentials saved to {kaggle_json}")
                return True

        print("Skipping download. Please set up Kaggle credentials and retry.")
        return False
    return True


def download_kaggle_dataset(dataset_slug: str, output_dir: Path):
    """Download a dataset from Kaggle using the API."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    print(f"\nDownloading: {dataset_slug}")
    print(f"Destination: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(dataset_slug, path=str(output_dir), unzip=True)
    print(f"Download complete!")


def organize_cnmc(raw_dir: Path, output_dir: Path):
    """
    Organize C-NMC 2019 dataset into ALL/ and Normal/ folders.

    C-NMC has: training_data/ with fold_0, fold_1, fold_2
    Each fold has: all/ and hem/ (normal) subdirectories
    """
    print("\nOrganizing C-NMC 2019 dataset...")

    all_dir = output_dir / "ALL"
    normal_dir = output_dir / "Normal"
    all_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)

    count_all, count_normal = 0, 0

    # Search for image files recursively
    for img_path in raw_dir.rglob("*"):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            continue

        # Determine class from parent directory name
        parent_lower = img_path.parent.name.lower()
        grandparent_lower = img_path.parent.parent.name.lower() if img_path.parent.parent else ""

        if "all" in parent_lower or "all" in grandparent_lower:
            dest = all_dir / f"cnmc_all_{count_all:05d}{img_path.suffix}"
            shutil.copy2(str(img_path), str(dest))
            count_all += 1
        elif "hem" in parent_lower or "normal" in parent_lower or "hem" in grandparent_lower:
            dest = normal_dir / f"cnmc_normal_{count_normal:05d}{img_path.suffix}"
            shutil.copy2(str(img_path), str(dest))
            count_normal += 1

    print(f"  ALL: {count_all} images")
    print(f"  Normal: {count_normal} images")
    return count_all + count_normal


def organize_aml(raw_dir: Path, output_dir: Path):
    """
    Organize AML Cytomorphology dataset.

    Map AML cell types to AML class, and normal cells to Normal class.
    AML-indicative types: BLA (blast), PMO (promyelocyte), MYB (myelocyte variant)
    Normal types: everything else (SEG, LYT, MON, EOS, BAS, etc.)
    """
    print("\nOrganizing AML Cytomorphology dataset...")

    aml_dir = output_dir / "AML"
    normal_dir = output_dir / "Normal"
    aml_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)

    # AML-indicative cell types (immature myeloid cells)
    aml_types = {
        "bla", "blast", "apl", "pmo", "promyelocyte", "myb", "myeloblast",
        "myo", "myelocyte", "mob", "monoblast", "pmb", "ksc", "mmz",
        "metamyelocyte", "ebo", "erythroblast", "ngb",
    }
    count_aml, count_normal = 0, 0

    for img_path in raw_dir.rglob("*"):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            continue

        parent_lower = img_path.parent.name.lower()

        if any(t in parent_lower for t in aml_types):
            dest = aml_dir / f"aml_{count_aml:05d}{img_path.suffix}"
            shutil.copy2(str(img_path), str(dest))
            count_aml += 1
        else:
            dest = normal_dir / f"aml_normal_{count_normal:05d}{img_path.suffix}"
            shutil.copy2(str(img_path), str(dest))
            count_normal += 1

    print(f"  AML: {count_aml} images")
    print(f"  Normal: {count_normal} images")
    return count_aml + count_normal


def organize_blood_cell_cancer(raw_dir: Path, output_dir: Path):
    """
    Organize Blood Cell Cancer Detection dataset.

    Expected structure: folders for different cell types including
    lymphoblast (ALL), myeloblast (AML), and normal cells.
    """
    print("\nOrganizing Blood Cell Cancer dataset...")

    all_dir = output_dir / "ALL"
    aml_dir = output_dir / "AML"
    normal_dir = output_dir / "Normal"
    all_dir.mkdir(parents=True, exist_ok=True)
    aml_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)

    # Mapping patterns to classes
    class_map = {
        "lymphoblast": "ALL",
        "all": "ALL",
        "myeloblast": "AML",
        "aml": "AML",
        "normal": "Normal",
        "benign": "Normal",
        "neutrophil": "Normal",
        "lymphocyte": "Normal",
        "monocyte": "Normal",
        "eosinophil": "Normal",
        "basophil": "Normal",
        "platelet": "Normal",
    }

    counts = {"ALL": 0, "AML": 0, "Normal": 0}

    for img_path in raw_dir.rglob("*"):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            continue

        parent_lower = img_path.parent.name.lower().replace("_", "").replace("-", "")

        target_class = None
        for pattern, cls in class_map.items():
            if pattern in parent_lower:
                target_class = cls
                break

        if target_class is None:
            continue

        dest_dir = output_dir / target_class
        idx = counts[target_class]
        dest = dest_dir / f"bcc_{target_class.lower()}_{idx:05d}{img_path.suffix}"
        shutil.copy2(str(img_path), str(dest))
        counts[target_class] += 1

    for cls, cnt in counts.items():
        print(f"  {cls}: {cnt} images")
    return sum(counts.values())


def organize_all_subtypes(raw_dir: Path, output_dir: Path):
    """
    Organize ALL Image Dataset (subtypes).

    Categories: Benign, Early Pre-B, Pre-B, Pro-B → ALL or Normal
    """
    print("\nOrganizing ALL Subtypes dataset...")

    all_dir = output_dir / "ALL"
    normal_dir = output_dir / "Normal"
    all_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)

    count_all, count_normal = 0, 0

    for img_path in raw_dir.rglob("*"):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            continue

        parent_lower = img_path.parent.name.lower()

        if "benign" in parent_lower or "normal" in parent_lower:
            dest = normal_dir / f"allsub_normal_{count_normal:05d}{img_path.suffix}"
            shutil.copy2(str(img_path), str(dest))
            count_normal += 1
        else:
            # All malignant subtypes (Early Pre-B, Pre-B, Pro-B)
            dest = all_dir / f"allsub_all_{count_all:05d}{img_path.suffix}"
            shutil.copy2(str(img_path), str(dest))
            count_all += 1

    print(f"  ALL: {count_all} images")
    print(f"  Normal: {count_normal} images")
    return count_all + count_normal


# Dataset registry
DATASETS = {
    "cnmc": {
        "name": "C-NMC 2019 (ALL/Normal, ~15K images)",
        "slug": "andrewmvd/leukemia-classification",
        "organizer": organize_cnmc,
    },
    "aml": {
        "name": "AML Cytomorphology (~18K images)",
        "slug": "gustaveroussy/aml-cytomorphology",
        "alt_slugs": [
            "nikhilsharma00/aml-cancer-image-dataset-of-blood-cell",
        ],
        "organizer": organize_aml,
    },
    "blood_cell": {
        "name": "Blood Cell Cancer Detection (~5K images)",
        "slug": "mohammadamireshraghi/blood-cell-cancer-all-4class",
        "alt_slugs": [
            "sajidsarker/blood-cell-images-for-cancer-detection",
        ],
        "organizer": organize_blood_cell_cancer,
    },
    "all_subtypes": {
        "name": "ALL Image Dataset (~3.2K images)",
        "slug": "mehradaria/leukemia",
        "alt_slugs": [
            "ambarish/breakhis-microscopy",
        ],
        "organizer": organize_all_subtypes,
    },
}


def download_and_organize(dataset_key: str, data_dir: Path):
    """Download and organize a single dataset."""
    info = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"Dataset: {info['name']}")
    print(f"{'='*60}")

    raw_dir = data_dir / "raw" / dataset_key
    organized_dir = data_dir

    # Download
    slug = info["slug"]
    try:
        download_kaggle_dataset(slug, raw_dir)
    except Exception as e:
        print(f"  Primary download failed: {e}")
        # Try alternative slugs
        if "alt_slugs" in info:
            for alt in info["alt_slugs"]:
                try:
                    print(f"  Trying alternative: {alt}")
                    download_kaggle_dataset(alt, raw_dir)
                    break
                except Exception as e2:
                    print(f"  Also failed: {e2}")
                    continue
            else:
                print(f"  All download attempts failed for {dataset_key}")
                return 0
        else:
            return 0

    # Organize
    count = info["organizer"](raw_dir, organized_dir)
    return count


def print_dataset_summary(data_dir: Path):
    """Print summary of organized dataset."""
    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print(f"{'='*60}")

    total = 0
    for class_name in ["ALL", "AML", "CML", "Normal"]:
        class_dir = data_dir / class_name
        if class_dir.exists():
            count = len(list(class_dir.glob("*")))
            print(f"  {class_name:10s}: {count:>6,} images")
            total += count
        else:
            print(f"  {class_name:10s}:      0 images")

    print(f"  {'─'*28}")
    print(f"  {'TOTAL':10s}: {total:>6,} images")
    print(f"\nData directory: {data_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare leukemia datasets from Kaggle"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data",
        help="Output directory for organized dataset"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=list(DATASETS.keys()),
        help="Download a specific dataset"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Download all available datasets"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available datasets"
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable datasets:")
        for key, info in DATASETS.items():
            print(f"  {key:15s} — {info['name']}")
            print(f"  {'':15s}   kaggle: {info['slug']}")
        return

    if not args.dataset and not args.all:
        parser.print_help()
        print("\n\nExamples:")
        print("  py -3.10 scripts/download_dataset.py --all")
        print("  py -3.10 scripts/download_dataset.py --dataset cnmc")
        print("  py -3.10 scripts/download_dataset.py --list")
        return

    if not check_kaggle_credentials():
        return

    data_dir = Path(args.output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0

    if args.all:
        for key in DATASETS:
            total_images += download_and_organize(key, data_dir)
    elif args.dataset:
        total_images += download_and_organize(args.dataset, data_dir)

    print_dataset_summary(data_dir)

    if total_images > 0:
        print(f"\n✅ Dataset ready! You can now train with:")
        print(f"   py -3.10 scripts/train.py --data_dir {args.output_dir} --epochs 50")
    else:
        print(f"\n⚠ No images were organized. Check your Kaggle credentials and network.")


if __name__ == "__main__":
    main()
