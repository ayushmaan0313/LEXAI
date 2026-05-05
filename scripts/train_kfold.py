"""
Stratified K-Fold Cross-Validation training for LEXAI.

Uses k=5 stratified folds to ensure:
  - Every image is validated exactly once
  - Class proportions are preserved in each fold
  - Robust performance estimate across the full dataset
  - Best model from the best fold is saved for deployment

Usage:
  python scripts/train_kfold.py --data_dir data --k 5 --epochs 30
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lexai.config import LEXAIConfig
from lexai.models.lexai_model import LEXAIModel
from lexai.data.dataset import (
    LeukemiaDataset, scan_directory, compute_class_weights,
)
from lexai.data.preprocessing import get_train_transform, get_inference_transform
from lexai.training.trainer import Trainer


def create_fold_loaders(
    all_paths, all_labels, train_idx, val_idx,
    config, batch_size, num_workers,
):
    train_paths = [all_paths[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    val_paths = [all_paths[i] for i in val_idx]
    val_labels = [all_labels[i] for i in val_idx]

    train_transform = get_train_transform(config.data)
    val_transform = get_inference_transform(config.data)

    train_ds = LeukemiaDataset(train_paths, train_labels, train_transform)
    val_ds = LeukemiaDataset(val_paths, val_labels, val_transform)

    class_counts = np.bincount(train_labels, minlength=config.data.num_classes)
    sample_weights = [1.0 / (class_counts[l] + 1e-6) for l in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_labels),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, train_labels


def main():
    parser = argparse.ArgumentParser(
        description="Stratified K-Fold CV Training for LEXAI"
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--k", type=int, default=5, help="Number of folds")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_vit", action="store_true")
    parser.add_argument("--no_gnn", action="store_true")
    args = parser.parse_args()

    config = LEXAIConfig()
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    if args.no_vit:
        config.cnn.use_vit = False
    if args.no_gnn:
        config.gnn.enabled = False

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    all_paths, all_labels = scan_directory(args.data_dir, config.data)
    if not all_paths:
        print(f"No images found in {args.data_dir}")
        sys.exit(1)

    all_labels_np = np.array(all_labels)

    print(f"\n{'='*60}")
    print(f"  LEXAI — Stratified {args.k}-Fold Cross-Validation")
    print(f"{'='*60}")
    print(f"  Dataset: {len(all_paths):,} images")
    print(f"  Classes: {config.data.class_names}")
    print(f"  Folds: {args.k} | Epochs: {args.epochs}")
    print(f"  Device: {device}")

    print(f"\n  Class distribution:")
    for i, name in enumerate(config.data.class_names):
        count = int((all_labels_np == i).sum())
        print(f"    {name:12s}: {count:>6} ({count/len(all_labels_np)*100:.1f}%)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)
    fold_results = []
    best_overall_acc = 0.0
    best_fold = -1

    total_start = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(np.arange(len(all_paths)), all_labels_np), 1
    ):
        fold_start = time.time()
        print(f"\n{'─'*60}")
        print(f"  FOLD {fold_idx}/{args.k}  "
              f"(train={len(train_idx):,} | val={len(val_idx):,})")
        print(f"{'─'*60}")

        train_loader, val_loader, train_labels = create_fold_loaders(
            all_paths, all_labels, train_idx, val_idx,
            config, args.batch_size, args.num_workers,
        )

        class_weights = compute_class_weights(train_labels, config.data.num_classes)

        model = LEXAIModel(config)
        fold_output = output_dir / f"fold_{fold_idx}"

        trainer = Trainer(
            model, config, device=device,
            output_dir=str(fold_output),
            class_weights=class_weights.to(device),
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

        trainer.train(train_loader, val_loader)

        best_ckpt = fold_output / "best_model.pth"
        if best_ckpt.exists():
            trainer.load_checkpoint(str(best_ckpt))

        _, val_metrics = trainer.validate(val_loader)
        fold_time = time.time() - fold_start

        fold_result = {
            "fold": fold_idx,
            "accuracy": val_metrics["accuracy"],
            "f1_macro": val_metrics.get("f1_macro", 0),
            "precision_macro": val_metrics.get("precision_macro", 0),
            "recall_macro": val_metrics.get("recall_macro", 0),
            "ece": val_metrics.get("ece", 0),
            "time_seconds": fold_time,
        }
        fold_results.append(fold_result)

        print(f"\n  Fold {fold_idx}: Acc={fold_result['accuracy']:.4f} "
              f"F1={fold_result['f1_macro']:.4f} ECE={fold_result['ece']:.4f}")

        if fold_result["accuracy"] > best_overall_acc:
            best_overall_acc = fold_result["accuracy"]
            best_fold = fold_idx
            if best_ckpt.exists():
                shutil.copy2(str(best_ckpt), str(output_dir / "best_model.pth"))
                print(f"  New best fold! -> {output_dir / 'best_model.pth'}")

    total_time = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"  CROSS-VALIDATION RESULTS ({args.k} Folds)")
    print(f"{'='*60}")

    accs = [r["accuracy"] for r in fold_results]
    f1s = [r["f1_macro"] for r in fold_results]
    eces = [r["ece"] for r in fold_results]

    print(f"\n  {'Metric':<12} {'Mean':>8} {'Std':>8}")
    print(f"  {'─'*30}")
    for name, vals in [("Accuracy", accs), ("F1", f1s), ("ECE", eces)]:
        arr = np.array(vals)
        print(f"  {name:<12} {arr.mean():>8.4f} {arr.std():>8.4f}")

    print(f"\n  Best fold: {best_fold} (acc={best_overall_acc:.4f})")
    print(f"  Total time: {total_time/3600:.1f} hours")

    results_path = output_dir / "cv_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "k": args.k, "best_fold": best_fold,
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "fold_results": fold_results,
        }, f, indent=2)
    print(f"  Results: {results_path}")


if __name__ == "__main__":
    main()
