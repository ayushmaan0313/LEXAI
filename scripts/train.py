"""Training entrypoint for LEXAI model."""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lexai.config import LEXAIConfig
from lexai.models.lexai_model import LEXAIModel
from lexai.data.dataset import create_data_loaders, compute_class_weights
from lexai.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train LEXAI Model")
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to data directory with class subdirectories "
             "(Normal/, ALL/, AML/, CML/)",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--finetune_lr", type=float, default=5e-5)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to CSV manifest (path,label). Overrides --data_dir scanning.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument("--no_vit", action="store_true", help="Disable ViT backbone")
    parser.add_argument("--no_gnn", action="store_true", help="Disable GNN pathway")
    parser.add_argument(
        "--stain_norm", action="store_true", help="Enable Macenko stain normalization"
    )
    parser.add_argument(
        "--no_amp", action="store_true", help="Disable mixed precision (AMP)"
    )
    args = parser.parse_args()

    config = LEXAIConfig()
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.finetune_lr = args.finetune_lr
    if args.no_vit:
        config.cnn.use_vit = False
    if args.no_gnn:
        config.gnn.enabled = False
    if args.stain_norm:
        config.data.use_stain_norm = True

    print(f"Config: {config.data.num_classes} classes {config.data.class_names}")
    print(f"Backbones: {config.cnn.backbone_names} (ViT: {config.cnn.use_vit})")
    print(f"GNN: {'enabled' if config.gnn.enabled else 'disabled'}")

    print(f"\nLoading data from: {args.data_dir}")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        config=config.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        manifest_csv=args.manifest,
    )

    train_labels = train_loader.dataset.labels
    class_weights = compute_class_weights(train_labels, config.data.num_classes)
    print(f"\nClass weights:")
    for i, name in enumerate(config.data.class_names):
        print(f"  {name:12s}: {class_weights[i]:.4f}")

    model = LEXAIModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel params: {total_params:,} (trainable: {trainable:,})")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)} ({vram:.1f} GB)")

    trainer = Trainer(
        model, config, device=device,
        output_dir=args.output_dir,
        class_weights=class_weights.to(device),
        use_amp=not args.no_amp,
    )

    if args.resume:
        print(f"Resuming from: {args.resume}")
        resumed_metrics = trainer.load_checkpoint(args.resume)

    history = trainer.train(train_loader, val_loader, test_loader=test_loader)

    print(f"\nCheckpoints saved to: {args.output_dir}/")
    print("  best_model.pth       — best validation accuracy")
    print("  calibrated_model.pth — temperature-calibrated")


if __name__ == "__main__":
    main()
