"""CNN Ensemble with learnable weighted fusion and temperature calibration."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from lexai.config import CNNConfig, DEFAULT_CONFIG
from lexai.models.cnn_backbone import CNNBackbone


class TemperatureScaling(nn.Module):
    """Learnable temperature for post-hoc confidence calibration."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-4)


class CNNEnsemble(nn.Module):
    """
    Ensemble of CNN backbones with learnable per-backbone fusion weights
    and temperature calibration.
    """

    def __init__(self, config: CNNConfig = None):
        super().__init__()
        self.config = config or DEFAULT_CONFIG.cnn

        active_names = [
            n for n in self.config.backbone_names
            if n != "vit" or self.config.use_vit
        ]
        self.backbone_names = active_names

        self.backbones = nn.ModuleDict()
        for name in self.backbone_names:
            self.backbones[name] = CNNBackbone(
                name=name,
                output_dim=self.config.global_feature_dim,
                pretrained=self.config.pretrained,
                dropout=self.config.dropout,
            )

        num_backbones = len(self.backbone_names)

        self.fusion_weights = nn.Parameter(torch.ones(num_backbones))

        self.fusion_proj = nn.Sequential(
            nn.Linear(self.config.global_feature_dim, self.config.global_feature_dim),
            nn.BatchNorm1d(self.config.global_feature_dim),
            nn.ReLU(inplace=True),
        )

        self.calibration = TemperatureScaling()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        backbone_feats: Dict[str, torch.Tensor] = {}
        feat_list: List[torch.Tensor] = []

        for name in self.backbone_names:
            feat = self.backbones[name](x)
            backbone_feats[name] = feat
            feat_list.append(feat)

        stacked = torch.stack(feat_list, dim=1)

        w = F.softmax(self.fusion_weights, dim=0)
        weighted = stacked * w.view(1, -1, 1)
        fused = weighted.sum(dim=1)

        global_features = self.fusion_proj(fused)

        return {
            "global_features": global_features,
            "backbone_features": backbone_feats,
            "attention_weights": w.unsqueeze(0).expand(x.size(0), -1),
        }

    def get_target_layers(self) -> Dict[str, nn.Module]:
        return {
            name: self.backbones[name].target_layer
            for name in self.backbone_names
            if self.backbones[name].has_spatial_target
        }

    def get_fusion_weights(self) -> Dict[str, float]:
        w = F.softmax(self.fusion_weights, dim=0).detach().cpu().tolist()
        return dict(zip(self.backbone_names, w))

    def freeze_backbones(self):
        frozen = 0
        for name in self.backbone_names:
            for p in self.backbones[name].get_freezable_params():
                p.requires_grad = False
                frozen += 1
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Froze {frozen} backbone param groups. "
              f"Trainable: {trainable:,} / {total:,}")

    def unfreeze_backbones(self):
        for p in self.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Unfroze all. Trainable: {trainable:,}")

    def calibration_parameters(self):
        return self.calibration.parameters()
