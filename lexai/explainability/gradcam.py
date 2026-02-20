"""Grad-CAM explainability for CNN pathway.

Generates visual heatmaps showing which image regions the CNN
focused on for its prediction.
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Hooks into a target convolutional layer to produce a heatmap 
    showing which spatial regions are most important for a specific class.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(
            self._save_activation
        )
        self._backward_hook = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, module, input, output):
        """Hook to capture forward activations."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to capture backward gradients."""
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: (1, 3, H, W) preprocessed image
            target_class: class index to explain (or None for predicted class)

        Returns:
            heatmap: (H, W) normalized heatmap in [0, 1]
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # Handle different output formats
        if isinstance(output, dict):
            logits = output.get("logits", None)
            if logits is None:
                logits = output.get("global_features", None)
        else:
            logits = output

        if logits is None or logits.dim() < 2:
            # Return uniform heatmap if can't compute
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            return np.ones((h, w), dtype=np.float32) * 0.5

        # Select target class
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            return np.ones((h, w), dtype=np.float32) * 0.5

        # Compute weights: global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)  # Only positive contributions

        # Resize to input spatial dimensions
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay_heatmap(
        self,
        heatmap: np.ndarray,
        image: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on the original image.

        Args:
            heatmap: (H, W) normalized heatmap
            image: (H, W, 3) BGR uint8 image
            alpha: overlay transparency
            colormap: OpenCV colormap

        Returns:
            overlay: (H, W, 3) BGR uint8 image with heatmap
        """
        # Resize heatmap to match image
        heatmap_resized = cv2.resize(
            heatmap, (image.shape[1], image.shape[0])
        )

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), colormap
        )

        # Overlay
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

        return overlay

    def release(self):
        """Remove hooks."""
        self._forward_hook.remove()
        self._backward_hook.remove()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


class MultiBackboneGradCAM:
    """
    Manages Grad-CAM for multiple CNN backbones in the ensemble.
    Generates per-backbone heatmaps and a combined heatmap.
    """

    def __init__(self, model: nn.Module, target_layers: Dict[str, nn.Module]):
        """
        Args:
            model: The full LEXAI model or CNN ensemble
            target_layers: Dict mapping backbone name → target conv layer
        """
        self.cams: Dict[str, GradCAM] = {}
        for name, layer in target_layers.items():
            self.cams[name] = GradCAM(model, layer)

    def generate_all(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate Grad-CAM heatmaps for all backbones.

        Returns:
            Dict mapping backbone name → heatmap, plus 'combined' key
        """
        heatmaps = {}
        for name, cam in self.cams.items():
            heatmaps[name] = cam.generate(input_tensor, target_class)

        # Combined heatmap: weighted average
        if heatmaps:
            combined = np.mean(list(heatmaps.values()), axis=0)
            heatmaps["combined"] = combined

        return heatmaps

    def release(self):
        """Remove all hooks."""
        for cam in self.cams.values():
            cam.release()
