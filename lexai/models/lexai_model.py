"""Full LEXAI model orchestrating dual-pathway inference with multi-task outputs."""

from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from lexai.config import LEXAIConfig, DEFAULT_CONFIG
from lexai.data.segmentation import CellSegmenter, CellInfo
from lexai.data.preprocessing import tensor_to_numpy
from lexai.models.cnn_ensemble import CNNEnsemble
from lexai.models.gnn_pathway import GNNPathway
from lexai.models.fusion import MultiModalFusion


class LEXAIModel(nn.Module):
    """
    LEXAI: Dual-pathway Leukemia Detection Model.

    Pipeline:
        Input Image
            ├→ CNN Ensemble (EfficientNet + ResNet50 + DenseNet121) → 512-dim global features
            └→ Cell Segmentation → Graph Construction → GNN → 256-dim spatial features
          ↓
        Multi-Modal Fusion (Cross-modal attention) → 512-dim fused features
          ↓
        Multi-Task Heads:
            1. Classification → ALL / AML / CML / Normal
            2. Spatial Pattern Score → [0, 1] clustering metric
            3. Cell Density → Blast percentage [0, 100]
    """

    def __init__(self, config: LEXAIConfig = None):
        super().__init__()
        self.config = config or DEFAULT_CONFIG

        # CNN Global Analysis Pathway
        self.cnn_ensemble = CNNEnsemble(self.config.cnn)

        # GNN Spatial Analysis Pathway
        self.gnn_pathway = GNNPathway(self.config.gnn)

        # Multi-Modal Fusion
        self.fusion = MultiModalFusion(
            self.config.cnn, self.config.gnn, self.config.fusion
        )

        # Cell Segmenter (not a nn.Module, used in preprocessing)
        self.segmenter = CellSegmenter()

        # === Multi-Task Output Heads ===

        fused_dim = self.config.fusion.fused_dim  # 512

        # 1. Classification Head
        self.classification_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, self.config.data.num_classes),
        )

        # 2. Cell Density / Blast Percentage Head
        self.density_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # Output in [0, 1], scale to [0, 100]
        )

    def forward(
        self,
        images: torch.Tensor,
        graph_data: Optional[Any] = None,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            images: (B, 3, H, W) preprocessed image batch
            graph_data: Optional PyG Data/Batch for GNN pathway.
                        If None, GNN features are zeros.
            return_features: If True, include intermediate features

        Returns:
            dict with:
                'logits': (B, num_classes) classification logits
                'probabilities': (B, num_classes) softmax probabilities
                'predicted_class': (B,) predicted class indices
                'spatial_score': (B, 1) spatial pattern score
                'blast_percentage': (B, 1) estimated blast percentage
                'cnn_attention_weights': (B, 3) backbone attention weights
        """
        device = images.device
        batch_size = images.size(0)

        # --- CNN Pathway ---
        cnn_output = self.cnn_ensemble(images)
        global_features = cnn_output["global_features"]  # (B, 512)

        # --- GNN Pathway ---
        if graph_data is not None:
            gnn_output = self.gnn_pathway(graph_data, return_attention=True)
            spatial_features = gnn_output["spatial_features"]  # (B, 256)
            spatial_score = gnn_output["spatial_score"]         # (B, 1)
        else:
            gnn_output = self.gnn_pathway.create_dummy_output(
                batch_size=batch_size, device=device
            )
            spatial_features = gnn_output["spatial_features"]
            spatial_score = gnn_output["spatial_score"]

        # --- Multi-Modal Fusion ---
        fused = self.fusion(global_features, spatial_features)  # (B, 512)

        # --- Multi-Task Heads ---
        logits = self.classification_head(fused)                 # (B, C)
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1)

        blast_pct = self.density_head(fused) * 100.0             # (B, 1) → [0, 100]

        result = {
            "logits": logits,
            "probabilities": probabilities,
            "predicted_class": predicted_class,
            "spatial_score": spatial_score,
            "blast_percentage": blast_pct,
            "cnn_attention_weights": cnn_output["attention_weights"],
        }

        if return_features:
            result["global_features"] = global_features
            result["spatial_features"] = spatial_features
            result["fused_features"] = fused
            result["backbone_features"] = cnn_output["backbone_features"]
            if "attention_weights" in gnn_output:
                result["gnn_attention_weights"] = gnn_output["attention_weights"]
            if "node_embeddings" in gnn_output:
                result["node_embeddings"] = gnn_output["node_embeddings"]

        return result

    def process_single_image(
        self,
        image_tensor: torch.Tensor,
        original_image: np.ndarray,
        device: torch.device = None,
    ) -> Dict[str, Any]:
        """
        High-level inference for a single image.
        Handles cell segmentation, graph construction, and full inference.

        Args:
            image_tensor: (1, 3, H, W) preprocessed tensor
            original_image: (H, W, 3) BGR numpy array for segmentation
            device: target device

        Returns:
            Full result dict including cell info
        """
        device = device or next(self.parameters()).device
        image_tensor = image_tensor.to(device)

        # Cell segmentation on original image
        cells, seg_mask = self.segmenter.segment(original_image)

        # Build cell graph
        graph_data = None
        cell_features = None
        if len(cells) >= self.config.gnn.min_cells:
            cell_features = self.segmenter.extract_cell_features(
                original_image, cells, self.config.gnn.node_feature_dim
            )
            graph_data = self.gnn_pathway.build_graph_from_cells(
                cells, cell_features, device=device
            )

        # Forward pass
        result = self.forward(
            image_tensor, graph_data=graph_data, return_features=True
        )

        # Add cell info to result
        result["cells"] = cells
        result["cell_count"] = len(cells)
        result["segmentation_mask"] = seg_mask
        result["heuristic_blast_pct"] = self.segmenter.compute_blast_percentage(cells)

        if graph_data is not None:
            result["graph_data"] = graph_data

        return result

    def get_class_name(self, class_idx: int) -> str:
        """Map class index to name."""
        return self.config.data.class_names[class_idx]
