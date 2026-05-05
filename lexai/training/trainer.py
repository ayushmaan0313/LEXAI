"""3-stage training loop for LEXAI model with temperature calibration."""

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from lexai.config import LEXAIConfig, DEFAULT_CONFIG
from lexai.models.lexai_model import LEXAIModel
from lexai.training.losses import MultiTaskLoss
from lexai.training.metrics import MetricsCalculator

try:
    from torch_geometric.data import Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class Trainer:
    """
    3-stage training for the LEXAI model.

    Stage 1: Freeze backbones, train classifiers + fusion + GNN
    Stage 2: Unfreeze all, low-LR end-to-end fine-tuning
    Stage 3: Temperature calibration on validation set (LBFGS)
    """

    def __init__(
        self,
        model: LEXAIModel,
        config: LEXAIConfig = None,
        device: torch.device = None,
        output_dir: str = "checkpoints",
        class_weights: torch.Tensor = None,
        use_amp: bool = True,
    ):
        self.config = config or DEFAULT_CONFIG
        self.tc = self.config.training

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        self.model = model.to(self.device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = MultiTaskLoss(
            classification_weight=self.tc.classification_weight,
            spatial_weight=self.tc.spatial_score_weight,
            density_weight=self.tc.density_weight,
            label_smoothing=self.tc.label_smoothing,
            class_weights=class_weights,
        )

        self.metrics = MetricsCalculator(self.config.data.class_names)

        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [], "val_ece": [],
        }
        self.best_val_acc = 0.0

        self.use_gnn = self.config.gnn.enabled and HAS_PYG
        if self.use_gnn:
            import cv2
            from lexai.data.segmentation import CellSegmenter
            self._cv2 = cv2
            self._segmenter = CellSegmenter()

    def _make_optimizer(self, lr: float) -> AdamW:
        params = [p for p in self.model.parameters() if p.requires_grad]
        return AdamW(params, lr=lr, weight_decay=self.tc.weight_decay)

    def _build_graph_batch(self, img_paths, device):
        """Build a PyG Batch of cell graphs from original images."""
        if not HAS_PYG:
            return None

        graphs = []
        for path in img_paths:
            img = self._cv2.imread(str(path))
            if img is None:
                continue
            cells, _ = self._segmenter.segment(img)
            if len(cells) < self.config.gnn.min_cells:
                continue
            cell_feats = self._segmenter.extract_cell_features(
                img, cells, self.config.gnn.node_feature_dim
            )
            graph = self.model.gnn_pathway.build_graph_from_cells(
                cells, cell_feats, device=device
            )
            if graph is not None:
                graphs.append(graph)

        if not graphs:
            return None
        return Batch.from_data_list(graphs)

    def train_epoch(
        self, train_loader: DataLoader, optimizer: AdamW
    ) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []
        num_batches = 0

        for batch in train_loader:
            images, labels, paths = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

            graph_data = None
            if self.use_gnn:
                try:
                    graph_data = self._build_graph_batch(paths, self.device)
                except Exception:
                    pass

            with autocast("cuda", enabled=self.use_amp):
                predictions = self.model(images, graph_data=graph_data)
                targets = {"labels": labels}
                loss_dict = self.criterion(predictions, targets)
                loss = loss_dict["total_loss"]

            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item()
            all_preds.extend(predictions["predicted_class"].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        acc = accuracy_score(all_labels, all_preds)
        return avg_loss, acc

    @torch.no_grad()
    def validate(
        self, val_loader: DataLoader, calibrate: bool = False
    ) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            images, labels, paths = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

            with autocast("cuda", enabled=self.use_amp):
                predictions = self.model(
                    images, graph_data=None, calibrate=calibrate
                )
                targets = {"labels": labels}
                loss_dict = self.criterion(predictions, targets)
            total_loss += loss_dict["total_loss"].item()
            num_batches += 1

            self.metrics.update(
                labels.cpu().numpy(),
                predictions["predicted_class"].cpu().numpy(),
                predictions["probabilities"].cpu().numpy(),
            )

        avg_loss = total_loss / max(num_batches, 1)
        metrics = self.metrics.compute()
        return avg_loss, metrics

    def _calibrate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Stage 3: Fit temperature scaling on the validation set."""
        print(f"\n{'='*55}")
        print("Stage 3 — Temperature scaling calibration")
        print("=" * 55)

        best_path = self.output_dir / "best_model.pth"
        if best_path.exists():
            self.load_checkpoint(str(best_path))
        self.model.eval()

        _, metrics_before = self.validate(val_loader, calibrate=False)
        ece_before = metrics_before.get("ece", 0.0)
        print(f"  ECE before calibration: {ece_before:.4f}")

        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                images, labels, _ = batch
                images = images.to(self.device)
                predictions = self.model(images, graph_data=None, calibrate=False)
                all_logits.append(predictions["logits"].cpu())
                all_labels.append(labels)

        logits_t = torch.cat(all_logits).to(self.device)
        labels_t = torch.cat(all_labels).to(self.device)

        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS(
            self.model.calibration_parameters(),
            lr=self.tc.calibration_lr,
            max_iter=50,
        )

        def closure():
            optimizer.zero_grad()
            scaled = self.model.cnn_ensemble.calibration(logits_t)
            loss = nll_criterion(scaled, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)

        _, metrics_after = self.validate(val_loader, calibrate=True)
        ece_after = metrics_after.get("ece", 0.0)
        T = self.model.cnn_ensemble.calibration.temperature.item()
        print(f"  Temperature learned: {T:.4f}")
        print(f"  ECE after calibration: {ece_after:.4f} (target < 0.05)")

        calib_path = self.output_dir / "calibrated_model.pth"
        self.save_checkpoint(calib_path, -1, metrics_after)
        print(f"  Saved calibrated model -> {calib_path}")

        return metrics_after

    def _log(
        self, epoch: int, total: int, tr_loss: float, tr_acc: float,
        val_loss: float, val_acc: float, val_ece: float,
    ):
        fw = self.model.get_fusion_weights()
        w_str = " | ".join(f"{k[:3].upper()}={v:.2f}" for k, v in fw.items())
        print(
            f"  Ep {epoch:3d}/{total}  "
            f"tLoss={tr_loss:.4f} tAcc={tr_acc:.4f}  "
            f"vLoss={val_loss:.4f} vAcc={val_acc:.4f}  "
            f"ECE={val_ece:.4f}  [{w_str}]"
        )

    def _record(
        self, tr_loss: float, tr_acc: float,
        val_loss: float, val_acc: float, val_ece: float,
    ):
        self.history["train_loss"].append(tr_loss)
        self.history["val_loss"].append(val_loss)
        self.history["train_acc"].append(tr_acc)
        self.history["val_acc"].append(val_acc)
        self.history["val_ece"].append(val_ece)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader = None,
    ) -> Dict:
        """
        Full 3-stage training loop.

        Stage 1: Freeze backbones, train heads + fusion
        Stage 2: Unfreeze all, low-LR fine-tuning
        Stage 3: Temperature calibration on val set
        """
        epochs = self.tc.epochs
        stage1_epochs = max(1, self.tc.freeze_backbone_epochs)
        stage2_epochs = max(1, epochs - stage1_epochs)
        patience = self.tc.patience

        print(f"\n{'='*55}")
        print("LEXAI 3-Stage Training")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {epochs} (Stage 1: {stage1_epochs}, Stage 2: {stage2_epochs})")
        print(f"  Stage 1 LR: {self.tc.learning_rate:.1e}")
        print(f"  Stage 2 LR: {self.tc.finetune_lr:.1e}")
        print(f"  Patience: {patience}")
        print(f"  GNN: {'enabled' if self.use_gnn else 'disabled (CNN-only)'}")
        print(f"  AMP: {'enabled (fp16)' if self.use_amp else 'disabled (fp32)'}")
        print("=" * 55)

        # ---- Stage 1: Frozen backbones ----
        print(f"\n{'='*55}")
        print(f"Stage 1 — Classifier + fusion training ({stage1_epochs} epochs)")
        print("=" * 55)
        self.model.freeze_backbones()
        optimizer = self._make_optimizer(self.tc.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=stage1_epochs)
        patience_counter = 0

        for epoch in range(1, stage1_epochs + 1):
            start = time.time()
            tr_loss, tr_acc = self.train_epoch(train_loader, optimizer)
            val_loss, val_metrics = self.validate(val_loader)
            scheduler.step()
            elapsed = time.time() - start

            val_acc = val_metrics["accuracy"]
            val_ece = val_metrics.get("ece", 0.0)
            self._log(epoch, stage1_epochs, tr_loss, tr_acc, val_loss, val_acc, val_ece)
            self._record(tr_loss, tr_acc, val_loss, val_acc, val_ece)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                patience_counter = 0
                self.save_checkpoint(
                    self.output_dir / "best_model.pth", epoch, val_metrics
                )
                print(f"  Saved best model (val_acc={val_acc:.4f}, ECE={val_ece:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n  Early stopping (Stage 1) at epoch {epoch}")
                    break

        # ---- Stage 2: Full fine-tuning ----
        print(f"\n{'='*55}")
        print(f"Stage 2 — Full fine-tuning ({stage2_epochs} epochs)")
        print("=" * 55)
        self.model.unfreeze_backbones()
        optimizer = self._make_optimizer(self.tc.finetune_lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=stage2_epochs)
        patience_counter = 0

        for epoch in range(1, stage2_epochs + 1):
            start = time.time()
            tr_loss, tr_acc = self.train_epoch(train_loader, optimizer)
            val_loss, val_metrics = self.validate(val_loader)
            scheduler.step()

            val_acc = val_metrics["accuracy"]
            val_ece = val_metrics.get("ece", 0.0)
            self._log(epoch, stage2_epochs, tr_loss, tr_acc, val_loss, val_acc, val_ece)
            self._record(tr_loss, tr_acc, val_loss, val_acc, val_ece)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                patience_counter = 0
                self.save_checkpoint(
                    self.output_dir / "best_model.pth", epoch, val_metrics
                )
                print(f"  Saved best model (val_acc={val_acc:.4f}, ECE={val_ece:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n  Early stopping (Stage 2) at epoch {epoch}")
                    break

        # ---- Stage 3: Temperature calibration ----
        calib_metrics = self._calibrate(val_loader)

        # ---- Final test evaluation ----
        if test_loader is not None:
            print(f"\n{'='*55}")
            print("Final Test Set Evaluation (calibrated)")
            print("=" * 55)
            _, test_metrics = self.validate(test_loader, calibrate=True)
            report = self.metrics.format_report(test_metrics)
            print(report)

        return self.history

    def save_checkpoint(self, path, epoch: int, metrics: Dict):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "best_val_acc": self.best_val_acc,
        }, path)

    def load_checkpoint(self, path: str) -> Dict:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        if "best_val_acc" in checkpoint:
            self.best_val_acc = checkpoint["best_val_acc"]
        print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
        return checkpoint.get("metrics", {})
