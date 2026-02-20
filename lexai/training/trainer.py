"""Training loop for LEXAI model — with anti-overfitting measures."""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from lexai.config import LEXAIConfig, DEFAULT_CONFIG
from lexai.models.lexai_model import LEXAIModel
from lexai.training.losses import MultiTaskLoss
from lexai.training.metrics import MetricsCalculator


class Trainer:
    """
    Training and evaluation loop for the LEXAI model.

    Anti-overfitting features:
    - Backbone freezing for initial epochs (transfer learning)
    - Class-weighted loss for imbalanced data
    - Gradient clipping
    - Cosine annealing with warmup
    - Early stopping with configurable patience
    - Checkpoint saving
    """

    def __init__(
        self,
        model: LEXAIModel,
        config: LEXAIConfig = None,
        device: torch.device = None,
        output_dir: str = "checkpoints",
        class_weights: torch.Tensor = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.tc = self.config.training

        # Device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self.model = model.to(self.device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Loss with class weights
        self.criterion = MultiTaskLoss(
            classification_weight=self.tc.classification_weight,
            spatial_weight=self.tc.spatial_score_weight,
            density_weight=self.tc.density_weight,
            label_smoothing=0.15,
            class_weights=class_weights,
        )

        # Optimizer — use different LR groups for backbone vs heads
        backbone_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        self.optimizer = AdamW([
            {"params": backbone_params, "lr": self.tc.learning_rate * 0.1},  # 10x lower LR for pretrained
            {"params": head_params, "lr": self.tc.learning_rate},
        ], weight_decay=self.tc.weight_decay)

        # Scheduler: linear warmup + cosine decay
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=self.tc.warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.tc.epochs - self.tc.warmup_epochs,
            eta_min=1e-7,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.tc.warmup_epochs],
        )

        # Metrics
        self.metrics = MetricsCalculator(self.config.data.class_names)

        # Training state
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        self._backbone_frozen = False

    def _freeze_backbones(self):
        """Freeze all pretrained backbone layers."""
        frozen_count = 0
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False
                frozen_count += 1
        self._backbone_frozen = True
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  🧊 Froze {frozen_count} backbone params. "
              f"Trainable: {trainable:,} / {total:,}")

    def _unfreeze_backbones(self):
        """Unfreeze all backbone layers for fine-tuning."""
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                param.requires_grad = True
        self._backbone_frozen = False
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  🔥 Unfroze backbones. "
              f"Trainable: {trainable:,} / {total:,}")

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            images, labels, _ = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward (no GNN graph for batch training — CNN-only path)
            predictions = self.model(images, graph_data=None)

            # Targets
            targets = {"labels": labels}

            # Loss
            loss_dict = self.criterion(predictions, targets)
            loss = loss_dict["total_loss"]

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_cls_loss += loss_dict["classification_loss"].item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_cls_loss = total_cls_loss / max(num_batches, 1)

        return {"total_loss": avg_loss, "classification_loss": avg_cls_loss}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        num_batches = 0

        for batch in val_loader:
            images, labels, _ = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

            predictions = self.model(images, graph_data=None)
            targets = {"labels": labels}

            loss_dict = self.criterion(predictions, targets)
            total_loss += loss_dict["total_loss"].item()
            num_batches += 1

            probs = predictions["probabilities"]
            preds = predictions["predicted_class"]

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        avg_loss = total_loss / max(num_batches, 1)

        # Compute metrics
        metrics = self.metrics.compute(
            np.array(all_preds),
            np.array(all_labels),
            np.array(all_probs),
        )

        return {"total_loss": avg_loss, **metrics}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
    ) -> Dict:
        """
        Run the full training loop.

        Returns:
            Training history dict
        """
        epochs = epochs or self.tc.epochs
        freeze_epochs = getattr(self.tc, "freeze_backbone_epochs", 0)

        print(f"\n{'='*60}")
        print(f"Training LEXAI Model")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {self.tc.batch_size}")
        print(f"Learning rate: {self.tc.learning_rate}")
        print(f"Weight decay: {self.tc.weight_decay}")
        print(f"Backbone freeze epochs: {freeze_epochs}")
        print(f"{'='*60}\n")

        # Freeze backbones for initial epochs
        if freeze_epochs > 0:
            self._freeze_backbones()

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Unfreeze backbones after freeze period
            if freeze_epochs > 0 and epoch == freeze_epochs + 1 and self._backbone_frozen:
                self._unfreeze_backbones()

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Step scheduler
            self.scheduler.step()

            elapsed = time.time() - start_time
            lr = self.optimizer.param_groups[0]["lr"]

            # Log
            self.history["train_loss"].append(train_metrics["total_loss"])
            self.history["val_loss"].append(val_metrics["total_loss"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])

            frozen_tag = " [backbone frozen]" if self._backbone_frozen else ""
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_metrics['total_loss']:.4f} | "
                f"Val Loss: {val_metrics['total_loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"LR: {lr:.2e} | "
                f"Time: {elapsed:.1f}s{frozen_tag}"
            )

            # Checkpoint
            if val_metrics["total_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["total_loss"]
                self.patience_counter = 0
                self.save_checkpoint(
                    self.output_dir / "best_model.pth",
                    epoch, val_metrics
                )
                print(f"  ✓ Saved best model (val_loss={self.best_val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.tc.patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        # Save final model
        self.save_checkpoint(
            self.output_dir / "final_model.pth",
            epochs, val_metrics
        )

        return self.history

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: Dict,
    ):
        """Save model checkpoint."""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
        return checkpoint.get("metrics", {})
