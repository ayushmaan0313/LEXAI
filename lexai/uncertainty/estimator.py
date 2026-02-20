"""Uncertainty estimation via Monte Carlo Dropout.

Runs multiple stochastic forward passes with dropout enabled
to estimate prediction variance and confidence intervals.
"""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


class UncertaintyEstimator:
    """
    MC Dropout uncertainty estimation.

    Enables dropout at inference time and runs N forward passes.
    The variance of predictions indicates model uncertainty:
    - Low variance → model is confident
    - High variance → model is uncertain (flag for review)
    """

    def __init__(self, model: nn.Module, n_samples: int = 20):
        self.model = model
        self.n_samples = n_samples

    def _enable_dropout(self):
        """Enable dropout layers during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def _disable_dropout(self):
        """Restore eval mode for dropout layers."""
        self.model.eval()

    @torch.no_grad()
    def estimate(
        self,
        images: torch.Tensor,
        graph_data=None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run MC Dropout inference.

        Args:
            images: (1, 3, H, W) preprocessed image
            graph_data: optional PyG Data for GNN pathway

        Returns:
            dict with:
                'mean_probabilities': (1, C) mean predicted probabilities
                'prediction_variance': (1, C) variance per class
                'confidence': (1,) confidence score [0, 1]
                'predictive_entropy': (1,) entropy of mean prediction
                'all_probabilities': (N, C) all sampled probabilities
                'confidence_interval_low': (1, C) 5th percentile
                'confidence_interval_high': (1, C) 95th percentile
        """
        self.model.eval()
        self._enable_dropout()

        all_probs = []

        for _ in range(self.n_samples):
            output = self.model(images, graph_data=graph_data)
            probs = output["probabilities"].cpu()  # (B, C)
            all_probs.append(probs)

        self._disable_dropout()

        # Stack: (N, B, C) → for B=1: (N, 1, C)
        all_probs = torch.stack(all_probs, dim=0)

        # Mean and variance
        mean_probs = all_probs.mean(dim=0)      # (B, C)
        var_probs = all_probs.var(dim=0)         # (B, C)

        # Confidence = 1 - normalized total variance
        total_var = var_probs.sum(dim=-1)         # (B,)
        max_possible_var = 0.25  # Max variance for binary uniform
        confidence = 1.0 - (total_var / max_possible_var).clamp(0, 1)

        # Predictive entropy
        entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)

        # Confidence intervals (5th and 95th percentiles)
        sorted_probs = all_probs.sort(dim=0)[0]
        idx_low = max(0, int(0.05 * self.n_samples))
        idx_high = min(self.n_samples - 1, int(0.95 * self.n_samples))
        ci_low = sorted_probs[idx_low]
        ci_high = sorted_probs[idx_high]

        return {
            "mean_probabilities": mean_probs,
            "prediction_variance": var_probs,
            "confidence": confidence,
            "predictive_entropy": entropy,
            "all_probabilities": all_probs.squeeze(1),  # (N, C)
            "confidence_interval_low": ci_low,
            "confidence_interval_high": ci_high,
        }

    def is_uncertain(
        self,
        uncertainty_result: Dict[str, torch.Tensor],
        confidence_threshold: float = 0.7,
        entropy_threshold: float = 1.0,
    ) -> bool:
        """
        Determine if the model's prediction should be flagged as uncertain.

        Args:
            uncertainty_result: output from estimate()
            confidence_threshold: minimum confidence to be "certain"
            entropy_threshold: maximum entropy to be "certain"

        Returns:
            True if the prediction is uncertain (should be reviewed)
        """
        conf = uncertainty_result["confidence"].item()
        entropy = uncertainty_result["predictive_entropy"].item()

        return conf < confidence_threshold or entropy > entropy_threshold
