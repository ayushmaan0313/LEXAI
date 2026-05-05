"""
Download and prepare leukemia datasets from Kaggle for 4-class classification.

Target structure:
  data/
    Normal/   — healthy blood cells
    ALL/      — Acute Lymphoblastic Leukemia
    AML/      — Acute Myeloid Leukemia
    CML/      — Chronic Myeloid Leukemia

Supported datasets:
  1. cnmc       — C-NMC 2019: ~15K images (ALL blast vs Normal)
  2. aml_tcia   — AML-Cytomorphology LMU: ~18K images (AML subtypes)
  3. blood_cell — Blood Cell Cancer Detection: ~5K images (multi-class)
  4. all_idb    — ALL-IDB2: ~260 images (ALL vs Normal, high quality)
  5. cml        — CML datasets (multiple sources attempted)

Usage:
  python scripts/download_dataset.py --all --output_dir data
  python scripts/download_dataset.py --dataset cnmc
  python scripts/download_dataset.py --dataset cml
  python scripts/download_dataset.py --list

Requires:
  pip install kaggle
  Kaggle API credentials at ~/.kaggle/kaggle.json
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def check_kaggle_credentials():
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

        username = input("Or enter your Kaggle username now (Enter to skip): ").strip()
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


def download_kaggle_dataset(slug: str, output_dir: Path):
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    print(f"  Downloading: {slug}")
    output_dir.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(slug, path=str(output_dir), unzip=True)
    print(f"  Download complete.")


def try_download(slugs: list, raw_dir: Path) -> bool:
    for slug in slugs:
        try:
            download_kaggle_dataset(slug, raw_dir)
            return True
        except Exception as e:
            print(f"  Failed ({slug}): {e}")
    return False


def _copy_images(src_dir: Path, dest_dir: Path, prefix: str, extensions=IMAGE_EXTENSIONS):
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = len(list(dest_dir.glob("*")))
    copied = 0
    for img in sorted(src_dir.rglob("*")):
        if img.is_file() and img.suffix.lower() in extensions:
            dest = dest_dir / f"{prefix}_{count:06d}{img.suffix}"
            shutil.copy2(str(img), str(dest))
            count += 1
            copied += 1
    return copied


# ── Organizer functions ──────────────────────────────────────────────


def organize_cnmc(raw_dir: Path, output_dir: Path) -> dict:
    """C-NMC 2019: ALL blast cells vs normal (HEM)."""
    print("\n  Organizing C-NMC 2019...")
    counts = {"ALL": 0, "Normal": 0}

    for img_path in raw_dir.rglob("*"):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        parent = img_path.parent.name.lower()
        grandparent = img_path.parent.parent.name.lower()

        if "all" in parent or "all" in grandparent:
            cls = "ALL"
        elif any(k in parent or k in grandparent for k in ("hem", "normal", "benign")):
            cls = "Normal"
        else:
            continue

        dest_dir = output_dir / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"cnmc_{cls.lower()}_{counts[cls]:06d}{img_path.suffix}"
        shutil.copy2(str(img_path), str(dest))
        counts[cls] += 1

    for cls, cnt in counts.items():
        print(f"    {cls}: {cnt}")
    return counts


def organize_aml_tcia(raw_dir: Path, output_dir: Path) -> dict:
    """AML-Cytomorphology LMU: blast cells → AML, mature cells → Normal."""
    print("\n  Organizing AML-Cytomorphology...")

    aml_types = {
        "bla", "blast", "apl", "pmo", "promyelocyte", "myb", "myeloblast",
        "myo", "myelocyte", "mob", "monoblast", "pmb", "ksc", "mmz",
        "metamyelocyte", "ebo", "erythroblast", "ngb",
    }
    counts = {"AML": 0, "Normal": 0}

    for img_path in raw_dir.rglob("*"):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        parent = img_path.parent.name.lower()

        if any(t in parent for t in aml_types):
            cls = "AML"
        else:
            cls = "Normal"

        dest_dir = output_dir / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"tcia_{cls.lower()}_{counts[cls]:06d}{img_path.suffix}"
        shutil.copy2(str(img_path), str(dest))
        counts[cls] += 1

    for cls, cnt in counts.items():
        print(f"    {cls}: {cnt}")
    return counts


def organize_blood_cell(raw_dir: Path, output_dir: Path) -> dict:
    """Blood Cell Cancer dataset — multi-class (ALL subtypes + benign)."""
    print("\n  Organizing Blood Cell Cancer dataset...")

    class_map = {
        "lymphoblast": "ALL", "all": "ALL",
        "early": "ALL", "pre": "ALL", "pro": "ALL",
        "malignant": "ALL",
        "myeloblast": "AML", "aml": "AML",
        "cml": "CML", "chronic": "CML",
        "normal": "Normal", "benign": "Normal",
        "neutrophil": "Normal", "lymphocyte": "Normal",
        "monocyte": "Normal", "eosinophil": "Normal",
        "basophil": "Normal", "platelet": "Normal",
        "hem": "Normal",
    }
    counts = {"ALL": 0, "AML": 0, "CML": 0, "Normal": 0}

    for img_path in raw_dir.rglob("*"):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        parent = img_path.parent.name.lower().replace("_", " ").replace("-", " ")

        target = None
        for pattern, cls in class_map.items():
            if pattern in parent:
                target = cls
                break

        if target is None:
            continue

        dest_dir = output_dir / target
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"bcc_{target.lower()}_{counts[target]:06d}{img_path.suffix}"
        shutil.copy2(str(img_path), str(dest))
        counts[target] += 1

    for cls, cnt in sorted(counts.items()):
        if cnt > 0:
            print(f"    {cls}: {cnt}")
    return counts


def organize_all_idb(raw_dir: Path, output_dir: Path) -> dict:
    """ALL-IDB2: high-quality ALL vs Normal (small but clean)."""
    print("\n  Organizing ALL-IDB2...")
    counts = {"ALL": 0, "Normal": 0}

    for img_path in raw_dir.rglob("*"):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        parent = img_path.parent.name.lower()

        if any(k in parent for k in ("all", "blast", "malignant", "positive", "1")):
            cls = "ALL"
        elif any(k in parent for k in ("normal", "benign", "healthy", "negative", "0", "hem")):
            cls = "Normal"
        else:
            continue

        dest_dir = output_dir / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"allidb_{cls.lower()}_{counts[cls]:06d}{img_path.suffix}"
        shutil.copy2(str(img_path), str(dest))
        counts[cls] += 1

    for cls, cnt in counts.items():
        if cnt > 0:
            print(f"    {cls}: {cnt}")
    return counts


def organize_cml(raw_dir: Path, output_dir: Path) -> dict:
    """CML datasets — various sources."""
    print("\n  Organizing CML dataset...")
    counts = {"CML": 0, "Normal": 0}

    cml_keywords = {"cml", "chronic", "cll"}
    normal_keywords = {"normal", "benign", "healthy", "hem", "negative"}

    for img_path in raw_dir.rglob("*"):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        parent = img_path.parent.name.lower().replace("_", " ").replace("-", " ")

        if any(k in parent for k in cml_keywords):
            cls = "CML"
        elif any(k in parent for k in normal_keywords):
            cls = "Normal"
        else:
            cls = "CML"

        dest_dir = output_dir / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"cml_{cls.lower()}_{counts[cls]:06d}{img_path.suffix}"
        shutil.copy2(str(img_path), str(dest))
        counts[cls] += 1

    for cls, cnt in counts.items():
        if cnt > 0:
            print(f"    {cls}: {cnt}")
    return counts


# ── Dataset registry ─────────────────────────────────────────────────

DATASETS = {
    "cnmc": {
        "name": "C-NMC 2019 — ALL vs Normal (~15K images)",
        "classes": ["ALL", "Normal"],
        "slugs": [
            "andrewmvd/leukemia-classification",
        ],
        "organizer": organize_cnmc,
    },
    "aml_tcia": {
        "name": "AML-Cytomorphology LMU — AML vs Normal (~18K images)",
        "classes": ["AML", "Normal"],
        "slugs": [
            "gustaveroussy/aml-cytomorphology",
            "nikhilsharma00/aml-cancer-image-dataset-of-blood-cell",
        ],
        "organizer": organize_aml_tcia,
    },
    "blood_cell": {
        "name": "Blood Cell Cancer Detection — multi-class (~5K images)",
        "classes": ["ALL", "AML", "Normal"],
        "slugs": [
            "mohammadamireshraghi/blood-cell-cancer-all-4class",
        ],
        "organizer": organize_blood_cell,
    },
    "all_idb": {
        "name": "ALL-IDB2 — ALL vs Normal (~260 images, high quality)",
        "classes": ["ALL", "Normal"],
        "slugs": [
            "nikhilsharma00/leukemia-classification",
        ],
        "organizer": organize_all_idb,
    },
    "cml": {
        "name": "CML datasets — Chronic Myeloid Leukemia",
        "classes": ["CML"],
        "slugs": [
            "avk256/cnmc-leukemia",
            "aadimator/leukemia-images-for-deep-learning",
            "paultimothymooney/blood-cells",
        ],
        "organizer": organize_cml,
    },
}


def download_and_organize(key: str, data_dir: Path) -> dict:
    info = DATASETS[key]
    print(f"\n{'='*60}")
    print(f"  {info['name']}")
    print(f"  Classes: {', '.join(info['classes'])}")
    print(f"{'='*60}")

    raw_dir = data_dir / "raw" / key

    if not try_download(info["slugs"], raw_dir):
        print(f"\n  All download attempts failed for '{key}'.")
        if key == "cml":
            print_cml_instructions(data_dir)
        return {}

    return info["organizer"](raw_dir, data_dir)


def print_cml_instructions(data_dir: Path):
    print()
    print("  CML data is scarce in public repositories.")
    print("  To add CML data manually:")
    print(f"    1. Create: {data_dir / 'CML'}/")
    print(f"    2. Place CML blood smear images (.jpg/.png) in that folder")
    print("    3. Sources to try:")
    print("       - Medical image repositories (TCIA, GDC)")
    print("       - Hospital collaborations")
    print("       - Published CML cytomorphology papers with supplementary data")
    print("    4. Run: python scripts/prepare_data.py --data_dir data")


def print_summary(data_dir: Path):
    print(f"\n{'='*60}")
    print("  DATASET SUMMARY")
    print(f"{'='*60}")
    total = 0
    for cls in ["Normal", "ALL", "AML", "CML"]:
        cls_dir = data_dir / cls
        if cls_dir.exists():
            count = sum(1 for f in cls_dir.iterdir()
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
            print(f"    {cls:10s}: {count:>6,} images")
            total += count
        else:
            print(f"    {cls:10s}:      0 images  (missing)")
    print(f"    {'─'*30}")
    print(f"    {'TOTAL':10s}: {total:>6,} images")

    missing = [c for c in ["Normal", "ALL", "AML", "CML"]
               if not (data_dir / c).exists() or not any((data_dir / c).iterdir())]
    if missing:
        print(f"\n  Missing classes: {', '.join(missing)}")
        print(f"  Download more: python scripts/download_dataset.py --list")

    print(f"\n  Next step:")
    print(f"    python scripts/prepare_data.py --data_dir {data_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and organize leukemia datasets from Kaggle (4-class)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data",
        help="Output directory for organized dataset",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=list(DATASETS.keys()),
        help="Download a specific dataset",
    )
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable datasets:")
        for key, info in DATASETS.items():
            print(f"  {key:15s} — {info['name']}")
            print(f"  {'':15s}   Provides: {', '.join(info['classes'])}")
        print("\nRecommended for full 4-class setup:")
        print("  python scripts/download_dataset.py --all")
        return

    if not args.dataset and not args.all:
        parser.print_help()
        print("\n\nExamples:")
        print("  python scripts/download_dataset.py --all")
        print("  python scripts/download_dataset.py --dataset cnmc")
        print("  python scripts/download_dataset.py --list")
        return

    if not check_kaggle_credentials():
        return

    data_dir = Path(args.output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        for key in DATASETS:
            download_and_organize(key, data_dir)
    elif args.dataset:
        download_and_organize(args.dataset, data_dir)

    print_summary(data_dir)


if __name__ == "__main__":
    main()
