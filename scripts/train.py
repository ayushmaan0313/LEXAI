"""Training entrypoint for LEXAI model."""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lexai.config import LEXAIConfig
from lexai.models.lexai_model import LEXAIModel
from lexai.data.dataset import create_data_loaders, compute_class_weights, LeukemiaDataset
from lexai.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train LEXAI Model")
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to data directory with class subdirectories"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from"
    )
    args = parser.parse_args()

    # Config
    config = LEXAIConfig()
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr

    # Data
    print(f"Loading data from: {args.data_dir}")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        config=config.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=True,
    )
    print(f"\nTrain: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)} | "
          f"Test: {len(test_loader.dataset)}")

    # Compute class weights for loss
    temp_ds = LeukemiaDataset(args.data_dir, config.data, split="val")
    class_weights = compute_class_weights(temp_ds.samples, config.data.num_classes)
    print(f"\nClass weights for loss:")
    for i, name in enumerate(config.data.class_names):
        print(f"  {name:20s}: {class_weights[i]:.4f}")

    # Model
    model = LEXAIModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel params: {total_params:,} (trainable: {trainable:,})")

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Clear GPU cache before training
    if device.type == "cuda":
        torch.cuda.empty_cache()
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)} ({vram:.1f} GB)")

    # Trainer with class weights
    trainer = Trainer(
        model, config, device=device,
        output_dir=args.output_dir,
        class_weights=class_weights.to(device),
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from: {args.resume}")
        resumed_metrics = trainer.load_checkpoint(args.resume)
        if resumed_metrics:
            trainer.best_val_loss = resumed_metrics.get("total_loss", float("inf"))
            print(f"  Best val_loss so far: {trainer.best_val_loss:.4f}")

    history = trainer.train(train_loader, val_loader, epochs=args.epochs)

    # Evaluate on test set
    print("\n--- Test Set Evaluation ---")
    test_metrics = trainer.validate(test_loader)
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    print(f"\nCheckpoints saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
