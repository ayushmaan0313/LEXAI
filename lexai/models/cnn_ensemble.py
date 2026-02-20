"""CNN Ensemble: fuses EfficientNet + ResNet50 + DenseNet121 into 512-dim global features."""

import torch
import torch.nn as nn
from typing import Dict, List

from lexai.config import CNNConfig, DEFAULT_CONFIG
from lexai.models.cnn_backbone import CNNBackbone


class CNNEnsemble(nn.Module):
    """
    Ensemble of three CNN backbones with learned attention-based fusion.

    Each backbone extracts features independently. Learned attention weights
    determine how much each backbone contributes to the final representation.
    """

    def __init__(self, config: CNNConfig = None):
        super().__init__()
        self.config = config or DEFAULT_CONFIG.cnn

        # Create backbone models
        self.backbones = nn.ModuleDict()
        for name in self.config.backbone_names:
            self.backbones[name] = CNNBackbone(
                name=name,
                output_dim=self.config.global_feature_dim,
                pretrained=self.config.pretrained,
                dropout=self.config.dropout,
            )

        num_backbones = len(self.config.backbone_names)

        # Attention mechanism for ensemble weighting
        self.attention = nn.Sequential(
            nn.Linear(
                self.config.global_feature_dim * num_backbones,
                num_backbones * 4,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(num_backbones * 4, num_backbones),
            nn.Softmax(dim=-1),
        )

        # Final projection after weighted fusion
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.config.global_feature_dim, self.config.global_feature_dim),
            nn.BatchNorm1d(self.config.global_feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) input image tensor

        Returns:
            dict with:
                'global_features': (B, 512) fused feature vector
                'backbone_features': dict of (B, 512) per-backbone features
                'attention_weights': (B, num_backbones) learned weights
        """
        # Extract features from each backbone
        backbone_feats: Dict[str, torch.Tensor] = {}
        feat_list: List[torch.Tensor] = []

        for name in self.config.backbone_names:
            feat = self.backbones[name](x)
            backbone_feats[name] = feat
            feat_list.append(feat)

        # Stack features: (B, num_backbones, dim)
        stacked = torch.stack(feat_list, dim=1)

        # Compute attention weights from concatenated features
        concat = torch.cat(feat_list, dim=-1)  # (B, num_backbones * dim)
        attn_weights = self.attention(concat)   # (B, num_backbones)

        # Weighted fusion
        weighted = stacked * attn_weights.unsqueeze(-1)  # (B, N, dim)
        fused = weighted.sum(dim=1)                       # (B, dim)

        # Final projection
        global_features = self.fusion_proj(fused)

        return {
            "global_features": global_features,
            "backbone_features": backbone_feats,
            "attention_weights": attn_weights,
        }

    def get_target_layers(self) -> Dict[str, nn.Module]:
        """Return the Grad-CAM target layers for each backbone."""
        return {
            name: self.backbones[name].target_layer
            for name in self.config.backbone_names
        }
