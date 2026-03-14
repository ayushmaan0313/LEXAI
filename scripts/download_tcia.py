"""
Download the AML-Cytomorphology_LMU dataset from The Cancer Imaging Archive (TCIA).

Dataset: 18,365 expert-labeled single-cell images from peripheral blood smears
  - 100 AML patients + 100 non-malignant controls
  - Munich University Hospital, 2014-2017
  - License: CC BY 3.0

Citation:
  Matek, C., Schwarz, S., Marr, C., & Spiekermann, K. (2019).
  A Single-cell Morphological Dataset of Leukocytes from AML Patients
  and Non-malignant Controls [Data set]. The Cancer Imaging Archive.
  https://doi.org/10.7937/tcia.2019.36f5o9ld

Usage:
  py scripts/download_tcia.py                          # Download + organize
  py scripts/download_tcia.py --output_dir data        # Custom output dir
  py scripts/download_tcia.py --skip_organize           # Download only
  py scripts/download_tcia.py --organize_only raw_dir   # Organize existing download
"""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError


# TCIA direct download URL (HTTPS fallback — may be slow for 11 GB)
TCIA_HTTPS_URL = (
    "https://faspex.cancerimagingarchive.net/aspera/faspex/public/package"
    "?context=eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb"
    "2FkX3BhY2thZ2UiLCJpZCI6IjczNSIsInBhc3Njb2RlIjoiYTczZTE1NzU1MjI5MzZkODR"
    "hZTY3MTcxMmU1YTg2YWY1ZTZlODI4MyIsInBhY2thZ2VfaWQiOiI3MzUiLCJlbWFpbCI6I"
    "mhlbHBAY2FuY2VyaW1hZ2luZ2FyY2hpdmUubmV0In0="
)

# Alternative: Kaggle mirrors of the same dataset
KAGGLE_SLUGS = [
    "gustaveroussy/aml-cytomorphology",
    "nikhilsharma00/aml-cancer-image-dataset-of-blood-cell",
]

# AML-indicative cell types in the TCIA dataset
# BLA/blast = blast cells, MYB = myeloblast, PMO = promyelocyte
# MYO = myelocyte (immature), MOB = monoblast (AML-M5)
# PMB = promyelocyte band, KSC = smudge/basket cell
# MMZ = metamyelocyte (immature), EBO = erythroblast (AML-M6)
# NGB = neutrophil band (immature, left shift indicator)
AML_CELL_TYPES = {
    "bla", "blast", "apl", "pmo", "promyelocyte", "myb", "myeloblast",
    "myo", "myelocyte", "mob", "monoblast", "pmb", "ksc", "mmz",
    "metamyelocyte", "ebo", "erythroblast", "ngb",
}


def download_progress(block_num, block_size, total_size):
    """Show download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded * 100.0 / total_size)
        mb_done = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  Downloading: {mb_done:.0f}/{mb_total:.0f} MB ({pct:.1f}%)")
    else:
        mb_done = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r  Downloading: {mb_done:.0f} MB")
    sys.stdout.flush()


def try_kaggle_download(raw_dir: Path) -> bool:
    """Try downloading from Kaggle as a fallback."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        for slug in KAGGLE_SLUGS:
            try:
                print(f"\n  Trying Kaggle: {slug}")
                raw_dir.mkdir(parents=True, exist_ok=True)
                api.dataset_download_files(slug, path=str(raw_dir), unzip=True)
                print(f"  ✅ Downloaded from Kaggle: {slug}")
                return True
            except Exception as e:
                print(f"  ❌ Kaggle {slug} failed: {e}")
                continue
    except ImportError:
        print("  Kaggle API not installed (pip install kaggle)")
    except Exception as e:
        print(f"  Kaggle authentication failed: {e}")

    return False


def try_https_download(raw_dir: Path) -> bool:
    """Try direct HTTPS download from TCIA."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "aml_cytomorphology_lmu.zip"

    print(f"\n  Attempting HTTPS download from TCIA (~11 GB)...")
    print(f"  This may take a while. If it fails, use manual download.\n")

    try:
        urlretrieve(TCIA_HTTPS_URL, str(zip_path), reporthook=download_progress)
        print("\n  Download complete. Extracting...")

        if zip_path.stat().st_size < 1_000_000:
            # Too small — likely an HTML error page
            print("  ⚠ Downloaded file is too small — probably a redirect page.")
            zip_path.unlink()
            return False

        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            zf.extractall(str(raw_dir))
        zip_path.unlink()
        print("  ✅ Extracted successfully.")
        return True

    except (URLError, OSError, zipfile.BadZipFile) as e:
        print(f"\n  ❌ HTTPS download failed: {e}")
        if zip_path.exists():
            zip_path.unlink()
        return False


def organize_aml_data(raw_dir: Path, output_dir: Path) -> dict:
    """
    Organize AML Cytomorphology images into AML/ and Normal/ directories.

    Cell type mapping:
      AML class  → BLA (blast), PMO (promyelocyte), MYB (myeloblast/myelocyte)
      Normal class → SEG, LYT, MON, EOS, BAS, and other non-malignant types
    """
    print("\nOrganizing AML-Cytomorphology_LMU dataset...")

    aml_dir = output_dir / "AML"
    normal_dir = output_dir / "Normal"
    aml_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)

    counts = {"AML": 0, "Normal": 0, "skipped": 0}

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for img_path in raw_dir.rglob("*"):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in image_extensions:
            continue

        # Determine class from parent directory name
        parent_lower = img_path.parent.name.lower()

        if any(t in parent_lower for t in AML_CELL_TYPES):
            dest = aml_dir / f"tcia_aml_{counts['AML']:05d}{img_path.suffix}"
            shutil.copy2(str(img_path), str(dest))
            counts["AML"] += 1
        else:
            dest = normal_dir / f"tcia_normal_{counts['Normal']:05d}{img_path.suffix}"
            shutil.copy2(str(img_path), str(dest))
            counts["Normal"] += 1

    return counts


def print_manual_instructions():
    """Print instructions for manual download."""
    print("\n" + "=" * 66)
    print("  MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 66)
    print()
    print("  Automatic download failed. Please download manually:")
    print()
    print("  1. Visit: https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/")
    print("  2. Click the 'Download (11gb)' button")
    print("     - You may need the IBM Aspera Connect plugin:")
    print("       https://www.ibm.com/products/aspera/downloads")
    print("  3. Extract the download into:")
    print("       data/raw/tcia_aml/")
    print("  4. Re-run this script with --organize_only:")
    print("       py scripts/download_tcia.py --organize_only data/raw/tcia_aml")
    print()
    print("  Alternative: Download from Kaggle (same dataset):")
    print("    pip install kaggle")
    print("    py scripts/download_dataset.py --dataset aml")
    print()
    print("=" * 66)


def main():
    parser = argparse.ArgumentParser(
        description="Download AML-Cytomorphology_LMU dataset from TCIA"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data",
        help="Output directory for organized dataset (default: data)"
    )
    parser.add_argument(
        "--raw_dir", type=str, default=None,
        help="Raw download directory (default: data/raw/tcia_aml)"
    )
    parser.add_argument(
        "--skip_organize", action="store_true",
        help="Download only, don't organize into class directories"
    )
    parser.add_argument(
        "--organize_only", type=str, default=None, metavar="RAW_DIR",
        help="Skip download, organize an existing raw directory"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir) if args.raw_dir else output_dir / "raw" / "tcia_aml"

    # --- Organize-only mode ---
    if args.organize_only:
        raw_path = Path(args.organize_only)
        if not raw_path.exists():
            print(f"Error: Directory not found: {raw_path}")
            sys.exit(1)

        counts = organize_aml_data(raw_path, output_dir)
        print(f"\n  AML:    {counts['AML']:>6,} images → {output_dir / 'AML'}")
        print(f"  Normal: {counts['Normal']:>6,} images → {output_dir / 'Normal'}")
        print(f"\n✅ Dataset organized! Ready for training.")
        return

    # --- Download ---
    print("=" * 66)
    print("  AML-Cytomorphology_LMU Dataset Download")
    print("  Source: The Cancer Imaging Archive (TCIA)")
    print("  Size: ~18,365 images (~11 GB)")
    print("  License: CC BY 3.0")
    print("=" * 66)

    downloaded = False

    # Try Kaggle first (more reliable than TCIA HTTPS)
    print("\n[1/2] Trying Kaggle download...")
    downloaded = try_kaggle_download(raw_dir)

    # Fall back to TCIA HTTPS
    if not downloaded:
        print("\n[2/2] Trying TCIA HTTPS download...")
        downloaded = try_https_download(raw_dir)

    if not downloaded:
        print_manual_instructions()
        sys.exit(1)

    # --- Organize ---
    if args.skip_organize:
        print(f"\n✅ Downloaded to: {raw_dir}")
        print(f"   Run with --organize_only {raw_dir} to organize later.")
        return

    counts = organize_aml_data(raw_dir, output_dir)

    print(f"\n{'=' * 50}")
    print(f"  DOWNLOAD SUMMARY")
    print(f"{'=' * 50}")
    print(f"  AML:    {counts['AML']:>6,} images → {output_dir / 'AML'}")
    print(f"  Normal: {counts['Normal']:>6,} images → {output_dir / 'Normal'}")
    print(f"\n✅ Dataset ready! Next steps:")
    print(f"   1. Reorganize data:  py scripts/reorganize_for_aml.py --data_dir {args.output_dir}")
    print(f"   2. Train the model:  py scripts/train.py --data_dir {args.output_dir} --preset aml --epochs 50")


if __name__ == "__main__":
    main()
