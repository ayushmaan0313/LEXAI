"""
Reorganize LEXAI dataset into proper leukemia subtype classes.

Creates the following class structure:
  data/
    ALL_Early_Pre_B/   — Early Pre-B ALL (from ALL-subtypes dataset)
    ALL_Pre_B/         — Pre-B ALL (from ALL-subtypes dataset)
    ALL_Pro_B/         — Pro-B ALL (from ALL-subtypes dataset)
    Benign/            — Normal/healthy cells (from ALL-subtypes + C-NMC hem)
    ALL_Blast/         — ALL blast cells from C-NMC (unsubtyped leukemia blast)

This gives 5 classes with real clinical labels.
"""

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def reorganize(data_dir: Path, raw_dir: Path):
    """Reorganize raw downloads into multi-class structure."""

    # Define class directories
    classes = {
        "ALL_Early_Pre_B": [],
        "ALL_Pre_B": [],
        "ALL_Pro_B": [],
        "Benign": [],
        "ALL_Blast": [],
    }

    for cls in classes:
        (data_dir / cls).mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # Source 1: ALL-subtypes dataset (labeled subtypes)
    # -------------------------------------------------------
    subtypes_dir = raw_dir / "all_subtypes" / "Original"
    if subtypes_dir.exists():
        mapping = {
            "Early": "ALL_Early_Pre_B",
            "Pre": "ALL_Pre_B",
            "Pro": "ALL_Pro_B",
            "Benign": "Benign",
        }
        for folder_name, target_class in mapping.items():
            src = subtypes_dir / folder_name
            if not src.exists():
                continue
            count = 0
            for img in sorted(src.iterdir()):
                if img.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                    dst = data_dir / target_class / f"subtypes_{folder_name.lower()}_{count:05d}{img.suffix}"
                    shutil.copy2(str(img), str(dst))
                    count += 1
            print(f"  ALL-Subtypes/{folder_name} → {target_class}: {count} images")
    else:
        print("  WARNING: ALL-subtypes raw data not found")

    # -------------------------------------------------------
    # Source 2: C-NMC dataset (ALL blast vs Normal)
    # -------------------------------------------------------
    cnmc_train = raw_dir / "cnmc" / "C-NMC_Leukemia" / "training_data"
    if cnmc_train.exists():
        # ALL blast cells (from fold_0, fold_1, fold_2 → "all")
        blast_count = 0
        normal_count = 0
        for fold in cnmc_train.iterdir():
            if not fold.is_dir():
                continue
            # ALL blasts
            all_dir = fold / "all"
            if all_dir.exists():
                for img in sorted(all_dir.iterdir()):
                    if img.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                        dst = data_dir / "ALL_Blast" / f"cnmc_blast_{blast_count:05d}{img.suffix}"
                        shutil.copy2(str(img), str(dst))
                        blast_count += 1
            # Normal/hem cells
            hem_dir = fold / "hem"
            if hem_dir.exists():
                for img in sorted(hem_dir.iterdir()):
                    if img.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                        dst = data_dir / "Benign" / f"cnmc_normal_{normal_count:05d}{img.suffix}"
                        shutil.copy2(str(img), str(dst))
                        normal_count += 1
        print(f"  C-NMC ALL blasts → ALL_Blast: {blast_count} images")
        print(f"  C-NMC hem/normal → Benign: {normal_count} images")
    else:
        print("  WARNING: C-NMC raw training data not found")

    # Also include C-NMC test/validation data
    for extra_name in ["testing_data", "validation_data"]:
        extra_dir = raw_dir / "cnmc" / "C-NMC_Leukemia" / extra_name
        if extra_dir.exists():
            extra_count = 0
            for img in extra_dir.rglob("*"):
                if img.is_file() and img.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                    # These are unlabeled test images - add to ALL_Blast (C-NMC test data)
                    dst = data_dir / "ALL_Blast" / f"cnmc_{extra_name}_{extra_count:05d}{img.suffix}"
                    shutil.copy2(str(img), str(dst))
                    extra_count += 1
            if extra_count > 0:
                print(f"  C-NMC {extra_name} → ALL_Blast: {extra_count} images")

    # -------------------------------------------------------
    # Source 3: Blood Cell Cancer dataset (same subtypes as source 1)
    # -------------------------------------------------------
    bcc_dir = raw_dir / "blood_cell" / "Blood cell Cancer [ALL]"
    if bcc_dir.exists():
        bcc_mapping = {
            "[Malignant] early Pre-B": "ALL_Early_Pre_B",
            "[Malignant] Pre-B": "ALL_Pre_B",
            "[Malignant] Pro-B": "ALL_Pro_B",
            "Benign": "Benign",
        }
        for folder_name, target_class in bcc_mapping.items():
            src = bcc_dir / folder_name
            if not src.exists():
                continue
            existing = len(list((data_dir / target_class).glob("*")))
            count = 0
            for img in sorted(src.iterdir()):
                if img.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                    dst = data_dir / target_class / f"bcc_{folder_name.lower().replace(' ','_').replace('[','').replace(']','')}_{existing+count:05d}{img.suffix}"
                    shutil.copy2(str(img), str(dst))
                    count += 1
            if count > 0:
                print(f"  BloodCellCancer/{folder_name} → {target_class}: {count} images")

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------
    print(f"\n{'='*50}")
    print("DATASET SUMMARY")
    print(f"{'='*50}")
    total = 0
    for cls in sorted(classes.keys()):
        cls_dir = data_dir / cls
        count = len(list(cls_dir.glob("*"))) if cls_dir.exists() else 0
        print(f"  {cls:20s}: {count:>6,} images")
        total += count
    print(f"  {'─'*35}")
    print(f"  {'TOTAL':20s}: {total:>6,} images")


if __name__ == "__main__":
    data_dir = PROJECT_ROOT / "data"
    raw_dir = data_dir / "raw"

    print("Reorganizing LEXAI dataset into cancer subtypes...\n")
    reorganize(data_dir, raw_dir)
    print(f"\nDone! Data ready at: {data_dir}")
