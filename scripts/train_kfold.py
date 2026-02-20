"""
Stratified K-Fold Cross-Validation training for LEXAI.

Uses k=5 stratified folds to ensure:
  - Every image is validated exactly once
  - Class proportions are preserved in each fold
  - Robust performance estimate across the full dataset
  - Best model from the best fold is saved for deployment

Usage:
    py -3.10 scripts/train_kfold.py --data_dir data --k 5 --epochs 50 --device cuda
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lexai.config import LEXAIConfig
from lexai.models.lexai_model import LEXAIModel
from lexai.data.dataset import (
    LeukemiaDataset, compute_class_weights, WeightedRandomSampler
)
from lexai.training.trainer import Trainer
from torch.utils.data import DataLoader, Subset


def create_fold_loaders(
    dataset: LeukemiaDataset,
    train_indices: list,
    val_indices: list,
    batch_size: int,
    num_workers: int,
    use_weighted_sampler: bool = True,
):
    """Create train and val DataLoaders for a single fold."""

    # Train dataset with augmentation
    train_ds = LeukemiaDataset(
        dataset.data_dir, dataset.config, split="train"
    )
    # Val dataset without augmentation
    val_ds = LeukemiaDataset(
        dataset.data_dir, dataset.config, split="val"
    )

    # Weighted sampler for balanced training
    sampler = None
    shuffle = True
    if use_weighted_sampler:
        class_counts = {}
        for i in train_indices:
            _, label = dataset.samples[i]
            class_counts[label] = class_counts.get(label, 0) + 1

        sample_weights = []
        for i in train_indices:
            _, label = dataset.samples[i]
            sample_weights.append(1.0 / class_counts[label])

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_indices),
            replacement=True,
        )
        shuffle = False

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

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(
        description="Stratified K-Fold CV Training for LEXAI"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to data directory with class subdirectories"
    )
    parser.add_argument("--k", type=int, default=5, help="Number of folds")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    # Config
    config = LEXAIConfig()
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)} ({vram:.1f} GB)")

    # Load full dataset (no augmentation, for splitting only)
    full_dataset = LeukemiaDataset(args.data_dir, config.data, split="val")
    all_labels = np.array([label for _, label in full_dataset.samples])
    all_indices = np.arange(len(full_dataset))

    # Class weights (computed on full dataset)
    class_weights = compute_class_weights(full_dataset.samples, config.data.num_classes)
    class_weights = class_weights.to(device)

    print(f"\n{'='*60}")
    print(f"  LEXAI — Stratified {args.k}-Fold Cross-Validation")
    print(f"{'='*60}")
    print(f"  Dataset:    {len(full_dataset):,} images")
    print(f"  Classes:    {config.data.class_names}")
    print(f"  Folds:      {args.k}")
    print(f"  Epochs:     {args.epochs} per fold")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Device:     {device}")

    # Class distribution
    print(f"\n  Class distribution:")
    for i, name in enumerate(config.data.class_names):
        count = int((all_labels == i).sum())
        print(f"    {name:20s}: {count:>6} ({count/len(all_labels)*100:.1f}%)")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)

    fold_results = []
    best_overall_loss = float("inf")
    best_fold = -1

    total_start = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(all_indices, all_labels), 1
    ):
        fold_start = time.time()

        print(f"\n{'─'*60}")
        print(f"  FOLD {fold_idx}/{args.k}")
        print(f"  Train: {len(train_idx):,} | Val: {len(val_idx):,}")
        print(f"{'─'*60}")

        # Show fold class distribution
        train_labels = all_labels[train_idx]
        val_labels = all_labels[val_idx]
        for i, name in enumerate(config.data.class_names):
            tc = int((train_labels == i).sum())
            vc = int((val_labels == i).sum())
            print(f"    {name:20s}: train={tc:>5}  val={vc:>4}")

        # Create data loaders for this fold
        train_loader, val_loader = create_fold_loaders(
            full_dataset,
            train_idx.tolist(),
            val_idx.tolist(),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # Fresh model for each fold
        model = LEXAIModel(config)

        # Trainer
        fold_output = output_dir / f"fold_{fold_idx}"
        fold_output.mkdir(parents=True, exist_ok=True)

        trainer = Trainer(
            model, config, device=device,
            output_dir=str(fold_output),
            class_weights=class_weights,
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Train
        history = trainer.train(train_loader, val_loader, epochs=args.epochs)

        # Final validation with best model
        best_ckpt = fold_output / "best_model.pth"
        if best_ckpt.exists():
            trainer.load_checkpoint(str(best_ckpt))

        final_metrics = trainer.validate(val_loader)
        fold_time = time.time() - fold_start

        fold_result = {
            "fold": fold_idx,
            "val_loss": final_metrics["total_loss"],
            "accuracy": final_metrics["accuracy"],
            "precision": final_metrics.get("precision", 0),
            "recall": final_metrics.get("recall", 0),
            "f1": final_metrics.get("f1", 0),
            "time_seconds": fold_time,
        }

        # Per-class metrics
        for name in config.data.class_names:
            p_key = f"precision_{name}"
            r_key = f"recall_{name}"
            if p_key in final_metrics:
                fold_result[p_key] = final_metrics[p_key]
            if r_key in final_metrics:
                fold_result[r_key] = final_metrics[r_key]

        fold_results.append(fold_result)

        print(f"\n  Fold {fold_idx} Results:")
        print(f"    Accuracy:  {fold_result['accuracy']:.4f}")
        print(f"    Precision: {fold_result['precision']:.4f}")
        print(f"    Recall:    {fold_result['recall']:.4f}")
        print(f"    F1:        {fold_result['f1']:.4f}")
        print(f"    Val Loss:  {fold_result['val_loss']:.4f}")
        print(f"    Time:      {fold_time:.0f}s ({fold_time/60:.1f} min)")

        # Track best fold
        if fold_result["val_loss"] < best_overall_loss:
            best_overall_loss = fold_result["val_loss"]
            best_fold = fold_idx
            # Copy best fold model as the deployment model
            import shutil
            deploy_path = output_dir / "best_model.pth"
            if best_ckpt.exists():
                shutil.copy2(str(best_ckpt), str(deploy_path))
                print(f"  ★ New best fold! Saved as {deploy_path}")

    # ---- Aggregate Results ----
    total_time = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"  CROSS-VALIDATION RESULTS ({args.k} Folds)")
    print(f"{'='*60}")

    accs = [r["accuracy"] for r in fold_results]
    precs = [r["precision"] for r in fold_results]
    recs = [r["recall"] for r in fold_results]
    f1s = [r["f1"] for r in fold_results]
    losses = [r["val_loss"] for r in fold_results]

    print(f"\n  {'Metric':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'─'*48}")
    for name, vals in [
        ("Accuracy", accs), ("Precision", precs),
        ("Recall", recs), ("F1", f1s), ("Val Loss", losses),
    ]:
        arr = np.array(vals)
        print(f"  {name:<12} {arr.mean():>8.4f} {arr.std():>8.4f} "
              f"{arr.min():>8.4f} {arr.max():>8.4f}")

    print(f"\n  Per-Fold Accuracy:")
    for r in fold_results:
        marker = " ★" if r["fold"] == best_fold else ""
        print(f"    Fold {r['fold']}: {r['accuracy']:.4f}  "
              f"(loss={r['val_loss']:.4f}){marker}")

    # Per-class aggregated metrics
    print(f"\n  Per-Class F1 (averaged across folds):")
    for class_name in config.data.class_names:
        p_key = f"precision_{class_name}"
        r_key = f"recall_{class_name}"
        p_vals = [r.get(p_key, 0) for r in fold_results]
        r_vals = [r.get(r_key, 0) for r in fold_results]
        # Compute F1 from mean P and R
        mean_p = np.mean(p_vals)
        mean_r = np.mean(r_vals)
        f1 = 2 * mean_p * mean_r / (mean_p + mean_r + 1e-8)
        print(f"    {class_name:20s}: P={mean_p:.4f}  R={mean_r:.4f}  F1={f1:.4f}")

    print(f"\n  Best fold: {best_fold} (val_loss={best_overall_loss:.4f})")
    print(f"  Best model: {output_dir / 'best_model.pth'}")
    print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f} hours)")

    # Save results JSON
    results_path = output_dir / "cv_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "k": args.k,
            "epochs_per_fold": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "best_fold": best_fold,
            "best_val_loss": best_overall_loss,
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_f1": float(np.mean(f1s)),
            "fold_results": fold_results,
        }, f, indent=2)
    print(f"  Results saved: {results_path}")


if __name__ == "__main__":
    main()
