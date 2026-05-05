"""Evaluation metrics for leukemia classification."""

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
)


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Compute Expected Calibration Error. Target: ECE < 0.05."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = predictions == labels

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc_in_bin = accuracies[mask].mean()
        conf_in_bin = confidences[mask].mean()
        ece += mask.mean() * abs(acc_in_bin - conf_in_bin)
    return float(ece)


class MetricsCalculator:
    """Compute classification metrics including FAR, EER, and ECE."""

    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or ["Normal", "ALL", "AML", "CML"]
        self.reset()

    def reset(self):
        self.all_labels = []
        self.all_preds = []
        self.all_probs = []

    def update(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray = None,
    ):
        self.all_labels.extend(labels.tolist())
        self.all_preds.extend(predictions.tolist())
        if probabilities is not None:
            self.all_probs.extend(probabilities.tolist())

    def compute(self) -> Dict[str, float]:
        labels = np.array(self.all_labels)
        preds = np.array(self.all_preds)
        probs = np.array(self.all_probs) if self.all_probs else None

        results = {}

        results["accuracy"] = accuracy_score(labels, preds)
        results["precision_macro"] = precision_score(
            labels, preds, average="macro", zero_division=0
        )
        results["recall_macro"] = recall_score(
            labels, preds, average="macro", zero_division=0
        )
        results["f1_macro"] = f1_score(
            labels, preds, average="macro", zero_division=0
        )

        precision_per = precision_score(labels, preds, average=None, zero_division=0)
        recall_per = recall_score(labels, preds, average=None, zero_division=0)
        for i, name in enumerate(self.class_names):
            if i < len(precision_per):
                results[f"precision_{name}"] = precision_per[i]
                results[f"recall_{name}"] = recall_per[i]

        cm = confusion_matrix(labels, preds)
        results["confusion_matrix"] = cm.tolist()

        if probs is not None and len(probs) > 0:
            probs_np = np.array(probs)
            if np.any(np.isnan(probs_np)):
                probs_np = np.nan_to_num(probs_np, nan=1.0 / len(self.class_names))
                row_sums = probs_np.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                probs_np = probs_np / row_sums
            results["ece"] = expected_calibration_error(probs_np, labels)
            far, eer = self._compute_far_eer(labels, probs_np)
            results["far"] = far
            results["eer"] = eer

        return results

    def _compute_far_eer(
        self, labels: np.ndarray, probs: np.ndarray
    ) -> Tuple[float, float]:
        num_classes = probs.shape[1] if probs.ndim > 1 else len(self.class_names)
        fars, eers = [], []

        for class_idx in range(num_classes):
            binary_labels = (labels == class_idx).astype(int)
            if probs.ndim > 1 and class_idx < probs.shape[1]:
                class_probs = probs[:, class_idx]
            else:
                continue

            if binary_labels.sum() == 0 or binary_labels.sum() == len(binary_labels):
                continue

            fpr, tpr, thresholds = roc_curve(binary_labels, class_probs)
            fnr = 1 - tpr

            pred_positive = (class_probs >= 0.5).astype(int)
            false_accepts = ((pred_positive == 1) & (binary_labels == 0)).sum()
            total_negative = (binary_labels == 0).sum()
            if total_negative > 0:
                fars.append(false_accepts / total_negative)

            eer_idx = np.argmin(np.abs(fpr - fnr))
            eers.append((fpr[eer_idx] + fnr[eer_idx]) / 2)

        far = np.mean(fars) if fars else 0.0
        eer = np.mean(eers) if eers else 0.0

        return far, eer

    def format_report(self, metrics: Dict = None) -> str:
        if metrics is None:
            metrics = self.compute()

        lines = [
            "=" * 55,
            "LEXAI Classification Report",
            "=" * 55,
            f"Accuracy:        {metrics['accuracy']:.4f}",
            f"Precision (avg): {metrics['precision_macro']:.4f}",
            f"Recall (avg):    {metrics['recall_macro']:.4f}",
            f"F1 Score (avg):  {metrics['f1_macro']:.4f}",
        ]

        if "ece" in metrics:
            lines.append(f"ECE:             {metrics['ece']:.4f}")
        if "far" in metrics:
            lines.append(f"FAR:             {metrics['far']:.4f}")
        if "eer" in metrics:
            lines.append(f"EER:             {metrics['eer']:.4f}")

        lines.append("-" * 55)
        lines.append("Per-Class Metrics:")
        for name in self.class_names:
            p = metrics.get(f"precision_{name}", 0)
            r = metrics.get(f"recall_{name}", 0)
            lines.append(f"  {name:8s}  P={p:.4f}  R={r:.4f}")

        lines.append("=" * 55)
        return "\n".join(lines)
