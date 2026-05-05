"""Full LEXAI model: CNN ensemble + GNN spatial pathway + multi-modal fusion."""

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
    Dual-pathway leukemia detection model.

    Pipeline:
        Input Image
            -> CNN Ensemble (EfficientNet + ResNet50 + DenseNet121 + ViT) -> 512-dim
            -> Cell Segmentation -> Graph -> GNN -> 256-dim
        Multi-Modal Fusion (cross-modal attention) -> 512-dim
        Multi-Task Heads:
            1. Classification -> Normal / ALL / AML / CML
            2. Spatial Pattern Score
            3. Blast Percentage
    """

    def __init__(self, config: LEXAIConfig = None):
        super().__init__()
        self.config = config or DEFAULT_CONFIG

        self.cnn_ensemble = CNNEnsemble(self.config.cnn)
        self.gnn_pathway = GNNPathway(self.config.gnn)
        self.fusion = MultiModalFusion(
            self.config.cnn, self.config.gnn, self.config.fusion
        )

        self.segmenter = CellSegmenter()

        fused_dim = self.config.fusion.fused_dim

        self.classification_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, self.config.data.num_classes),
        )

        self.density_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        images: torch.Tensor,
        graph_data=None,
        return_features: bool = False,
        calibrate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        device = images.device
        batch_size = images.size(0)

        cnn_output = self.cnn_ensemble(images)
        global_features = cnn_output["global_features"]

        if graph_data is not None:
            gnn_output = self.gnn_pathway(graph_data, return_attention=True)
            spatial_features = gnn_output["spatial_features"]
            spatial_score = gnn_output["spatial_score"]
        else:
            gnn_output = self.gnn_pathway.create_dummy_output(
                batch_size=batch_size, device=device
            )
            spatial_features = gnn_output["spatial_features"]
            spatial_score = gnn_output["spatial_score"]

        fused = self.fusion(global_features, spatial_features)

        logits = self.classification_head(fused)
        if calibrate:
            logits = self.cnn_ensemble.calibration(logits)
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1)

        blast_pct = self.density_head(fused) * 100.0

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
        calibrate: bool = True,
    ) -> Dict[str, Any]:
        device = device or next(self.parameters()).device
        image_tensor = image_tensor.to(device)

        cells, seg_mask = self.segmenter.segment(original_image)

        graph_data = None
        if len(cells) >= self.config.gnn.min_cells:
            cell_features = self.segmenter.extract_cell_features(
                original_image, cells, self.config.gnn.node_feature_dim
            )
            graph_data = self.gnn_pathway.build_graph_from_cells(
                cells, cell_features, device=device
            )

        result = self.forward(
            image_tensor, graph_data=graph_data,
            return_features=True, calibrate=calibrate
        )

        result["cells"] = cells
        result["cell_count"] = len(cells)
        result["segmentation_mask"] = seg_mask
        result["heuristic_blast_pct"] = self.segmenter.compute_blast_percentage(cells)

        if graph_data is not None:
            result["graph_data"] = graph_data

        return result

    def get_class_name(self, class_idx: int) -> str:
        return self.config.data.class_names[class_idx]

    def freeze_backbones(self):
        self.cnn_ensemble.freeze_backbones()

    def unfreeze_backbones(self):
        self.cnn_ensemble.unfreeze_backbones()

    def get_fusion_weights(self) -> Dict[str, float]:
        return self.cnn_ensemble.get_fusion_weights()

    def calibration_parameters(self):
        return self.cnn_ensemble.calibration_parameters()
