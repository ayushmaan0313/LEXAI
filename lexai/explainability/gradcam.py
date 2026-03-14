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
        self._activation = None

        # Only a FORWARD hook — no backward hook needed
        self._forward_hook = target_layer.register_forward_hook(
            self._save_activation
        )

    def _save_activation(self, module, input, output):
        """Capture activations and keep them in the graph for autograd."""
        self._activation = output
        if output.requires_grad:
            output.retain_grad()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Uses torch.autograd.grad instead of backward hooks to avoid
        conflicts with inplace operations in backbone networks.

        Args:
            input_tensor: (1, 3, H, W) preprocessed image
            target_class: class index to explain (or None for predicted class)

        Returns:
            heatmap: (H, W) normalized heatmap in [0, 1]
        """
        self.model.eval()
        self._activation = None

        # Forward pass (need gradients for the activations)
        output = self.model(input_tensor)

        # Handle different output formats
        if isinstance(output, dict):
            logits = output.get("logits", None)
            if logits is None:
                logits = output.get("global_features", None)
        else:
            logits = output

        if logits is None or logits.dim() < 2:
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            return np.ones((h, w), dtype=np.float32) * 0.5

        # Select target class
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        score = logits[0, target_class]

        # Compute gradients w.r.t activations using autograd.grad
        # This avoids backward hooks entirely
        if self._activation is None or not self._activation.requires_grad:
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            return np.ones((h, w), dtype=np.float32) * 0.5

        grads = torch.autograd.grad(
            score, self._activation,
            retain_graph=True, create_graph=False
        )[0]

        # Compute weights: global average pooling of gradients
        weights = grads.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        activations = self._activation.detach()
        cam = (weights.detach() * activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
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
