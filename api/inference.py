"""Inference pipeline for LEXAI — orchestrates model loading and prediction."""

import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from lexai.config import LEXAIConfig, DEFAULT_CONFIG
from lexai.data.preprocessing import preprocess_image, get_inference_transform, tensor_to_numpy
from lexai.data.segmentation import CellSegmenter
from lexai.models.lexai_model import LEXAIModel
from lexai.explainability.gradcam import GradCAM, MultiBackboneGradCAM
from lexai.explainability.gnn_explain import GNNExplainer
from lexai.uncertainty.estimator import UncertaintyEstimator


class InferencePipeline:
    """
    End-to-end inference pipeline.

    Loads the LEXAI model, processes an input image through both
    CNN and GNN pathways, generates explainability outputs, and
    estimates uncertainty.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config: LEXAIConfig = None,
        device: str = "auto",
    ):
        self.config = config or DEFAULT_CONFIG
        self.device = self._get_device(device)

        # Initialize model
        self.model = LEXAIModel(self.config)

        # Load checkpoint if available
        self.model_loaded = False
        if checkpoint_path and Path(checkpoint_path).exists():
            self._load_checkpoint(checkpoint_path)

        self.model.to(self.device)
        self.model.eval()

        # Explainability (initialized lazily)
        self._gradcam = None
        self._uncertainty = None

        # Cell segmenter + inference transform for per-cell classification
        self._segmenter = CellSegmenter()
        self._inference_transform = get_inference_transform(self.config.data)

    def _get_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model_loaded = True

    @property
    def gradcam(self) -> MultiBackboneGradCAM:
        if self._gradcam is None:
            target_layers = self.model.cnn_ensemble.get_target_layers()
            self._gradcam = MultiBackboneGradCAM(self.model, target_layers)
        return self._gradcam

    @property
    def uncertainty_estimator(self) -> UncertaintyEstimator:
        if self._uncertainty is None:
            self._uncertainty = UncertaintyEstimator(
                self.model,
                n_samples=self.config.training.mc_dropout_samples,
            )
        return self._uncertainty

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        """
        Full analysis pipeline for a single image.

        Args:
            image: PIL Image

        Returns:
            Complete result dict with classification, explainability,
            and uncertainty data
        """
        # Preprocess
        image_tensor = preprocess_image(image, self.config.data).to(self.device)
        original_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Main inference with cell segmentation
        with torch.no_grad():
            result = self.model.process_single_image(
                image_tensor, original_bgr, device=self.device
            )

        # Get predicted class info
        pred_idx = result["predicted_class"].item()
        pred_class = self.model.get_class_name(pred_idx)
        probs = result["probabilities"][0].cpu().numpy()

        # Probabilities dict
        prob_dict = {
            name: float(probs[i])
            for i, name in enumerate(self.config.data.class_names)
        }

        # --- Per-cell classification for full-field images ---
        cells = result.get("cells", [])
        cell_analyses = []
        if len(cells) >= 3:
            cell_analyses = self._classify_cells(original_bgr, cells)
            # Override prediction with aggregated per-cell vote
            if cell_analyses:
                agg_probs = np.zeros(len(self.config.data.class_names))
                for ca in cell_analyses:
                    for i, name in enumerate(self.config.data.class_names):
                        agg_probs[i] += ca["probabilities"].get(name, 0.0)
                agg_probs /= len(cell_analyses)
                pred_idx = int(np.argmax(agg_probs))
                pred_class = self.config.data.class_names[pred_idx]
                prob_dict = {
                    name: float(agg_probs[i])
                    for i, name in enumerate(self.config.data.class_names)
                }

        # --- Explainability ---

        # Grad-CAM (needs gradients enabled)
        try:
            with torch.enable_grad():
                heatmaps = self.gradcam.generate_all(
                    image_tensor.requires_grad_(True), pred_idx
                )
            combined_heatmap = heatmaps.get("combined", None)
            if combined_heatmap is not None:
                gradcam_overlay = self.gradcam.cams[
                    self.config.cnn.backbone_names[0]
                ].overlay_heatmap(combined_heatmap, original_bgr)
                gradcam_b64 = self._encode_image(gradcam_overlay)
            else:
                gradcam_b64 = None
        except Exception as e:
            import traceback
            print(f"[LEXAI] GradCAM failed: {e}")
            traceback.print_exc()
            gradcam_b64 = None

        # GNN graph visualization
        gnn_viz_b64 = None
        try:
            cells = result.get("cells", [])
            if cells and "graph_data" in result:
                graph_data = result["graph_data"]
                node_embeddings = result.get("node_embeddings", None)

                node_importance = None
                if node_embeddings is not None:
                    node_importance = GNNExplainer.compute_node_importance(
                        node_embeddings
                    )

                attn_weights = None
                if "gnn_attention_weights" in result:
                    attn_weights = GNNExplainer.extract_attention_weights(result)

                gnn_viz = GNNExplainer.visualize_cell_graph(
                    original_bgr, cells,
                    edge_index=graph_data.edge_index,
                    attention_weights=attn_weights,
                    node_importance=node_importance,
                )
                gnn_viz_b64 = self._encode_image(gnn_viz)
        except Exception as e:
            import traceback
            print(f"[LEXAI] GNN visualization failed: {e}")
            traceback.print_exc()
            gnn_viz_b64 = None

        # --- Uncertainty ---
        try:
            unc_result = self.uncertainty_estimator.estimate(
                image_tensor,
                graph_data=result.get("graph_data", None),
            )
            confidence = float(unc_result["confidence"].item())
            is_uncertain = self.uncertainty_estimator.is_uncertain(unc_result)

            var_dict = {
                name: float(unc_result["prediction_variance"][0, i])
                for i, name in enumerate(self.config.data.class_names)
            }
            ci_low_dict = {
                name: float(unc_result["confidence_interval_low"][0, i])
                for i, name in enumerate(self.config.data.class_names)
            }
            ci_high_dict = {
                name: float(unc_result["confidence_interval_high"][0, i])
                for i, name in enumerate(self.config.data.class_names)
            }
        except Exception as e:
            import traceback
            print(f"[LEXAI] Uncertainty estimation failed: {e}")
            traceback.print_exc()
            confidence = float(probs.max())
            is_uncertain = probs.max() < 0.7
            var_dict = {}
            ci_low_dict = {}
            ci_high_dict = {}

        # CNN backbone weights
        backbone_weights = {}
        if "cnn_attention_weights" in result:
            attn = result["cnn_attention_weights"][0].cpu().numpy()
            for i, name in enumerate(self.config.cnn.backbone_names):
                backbone_weights[name] = float(attn[i])

        return {
            "predicted_class": pred_class,
            "class_index": pred_idx,
            "probabilities": prob_dict,
            "spatial_pattern_score": float(
                result["spatial_score"][0].item()
            ),
            "cell_count": result.get("cell_count", 0),
            "blast_percentage": float(
                result["blast_percentage"][0].item()
            ),
            "confidence": confidence,
            "is_uncertain": is_uncertain,
            "prediction_variance": var_dict,
            "confidence_interval_low": ci_low_dict,
            "confidence_interval_high": ci_high_dict,
            "gradcam_heatmap": gradcam_b64,
            "gnn_graph_visualization": gnn_viz_b64,
            "cnn_backbone_weights": backbone_weights,
            "cell_analysis": cell_analyses,
        }

    def _classify_cells(
        self,
        image_bgr: np.ndarray,
        cells,
    ) -> List[Dict[str, Any]]:
        """
        Classify each segmented cell individually.

        Crops each cell from the full image, pads to square,
        resizes to model input size, and runs through CNN.

        Returns:
            List of per-cell analysis dicts with class, probs, bbox.
        """
        results = []
        h, w = image_bgr.shape[:2]

        for cell in cells:
            x, y, bw, bh = cell.bbox

            # Pad crop to square with 20% margin
            size = max(bw, bh)
            margin = int(size * 0.2)
            size_padded = size + 2 * margin

            cx, cy = cell.centroid
            x1 = max(0, cx - size_padded // 2)
            y1 = max(0, cy - size_padded // 2)
            x2 = min(w, x1 + size_padded)
            y2 = min(h, y1 + size_padded)

            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Convert BGR crop to PIL RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(crop_rgb)

            # Preprocess and classify
            tensor = self._inference_transform(pil_crop).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(tensor, graph_data=None)

            probs = output["probabilities"][0].cpu().numpy()
            pred_idx = int(probs.argmax())
            pred_class = self.config.data.class_names[pred_idx]

            results.append({
                "cell_id": len(results),
                "predicted_class": pred_class,
                "confidence": float(probs[pred_idx]),
                "probabilities": {
                    name: float(probs[i])
                    for i, name in enumerate(self.config.data.class_names)
                },
                "bbox": {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)},
                "centroid": {"x": int(cell.centroid[0]), "y": int(cell.centroid[1])},
            })

        return results

    @staticmethod
    def _encode_image(bgr_image: np.ndarray) -> str:
        """Encode a BGR image as base64 PNG string."""
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        buffer = io.BytesIO()
        pil.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
