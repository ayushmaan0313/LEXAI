"""Multi-task loss functions for LEXAI training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """
    Weighted multi-task loss combining:
    1. Classification: Cross-Entropy Loss
    2. Spatial Pattern Score: MSE Loss
    3. Cell Density (Blast %): MSE Loss
    
    Supports learnable task weights (uncertainty-based weighting)
    or fixed weights.
    """

    def __init__(
        self,
        classification_weight: float = 1.0,
        spatial_weight: float = 0.3,
        density_weight: float = 0.3,
        learnable_weights: bool = False,
        num_classes: int = 5,
        label_smoothing: float = 0.1,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing

        if learnable_weights:
            # Learnable log-variance weights (Kendall et al., 2018)
            self.log_vars = nn.Parameter(torch.zeros(3))
            self.fixed_weights = None
        else:
            self.log_vars = None
            self.fixed_weights = [
                classification_weight,
                spatial_weight,
                density_weight,
            ]

        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        predictions: dict,
        targets: dict,
    ) -> dict:
        """
        Compute multi-task loss.

        Args:
            predictions: dict from LEXAI model forward pass containing
                'logits', 'spatial_score', 'blast_percentage'
            targets: dict containing
                'labels': (B,) class indices
                'spatial_score': (B, 1) optional, defaults to 0
                'blast_percentage': (B, 1) optional, defaults to 0

        Returns:
            dict with 'total_loss', 'classification_loss',
            'spatial_loss', 'density_loss', and optional 'task_weights'
        """
        # Classification loss
        cls_loss = self.ce_loss(predictions["logits"], targets["labels"])

        # Spatial score loss (if target available)
        spatial_target = targets.get(
            "spatial_score",
            torch.zeros_like(predictions["spatial_score"])
        )
        spatial_loss = self.mse_loss(predictions["spatial_score"], spatial_target)

        # Density / blast percentage loss (if target available)
        density_target = targets.get(
            "blast_percentage",
            torch.zeros_like(predictions["blast_percentage"])
        )
        density_loss = self.mse_loss(
            predictions["blast_percentage"], density_target
        )

        # Weighted combination
        if self.log_vars is not None:
            # Uncertainty-based weighting
            precision0 = torch.exp(-self.log_vars[0])
            precision1 = torch.exp(-self.log_vars[1])
            precision2 = torch.exp(-self.log_vars[2])

            total_loss = (
                precision0 * cls_loss + self.log_vars[0]
                + precision1 * spatial_loss + self.log_vars[1]
                + precision2 * density_loss + self.log_vars[2]
            )
            task_weights = torch.exp(-self.log_vars).detach()
        else:
            total_loss = (
                self.fixed_weights[0] * cls_loss
                + self.fixed_weights[1] * spatial_loss
                + self.fixed_weights[2] * density_loss
            )
            task_weights = torch.tensor(self.fixed_weights)

        return {
            "total_loss": total_loss,
            "classification_loss": cls_loss,
            "spatial_loss": spatial_loss,
            "density_loss": density_loss,
            "task_weights": task_weights,
        }
