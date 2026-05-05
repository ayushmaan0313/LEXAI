"""CNN backbone wrappers for EfficientNet, ResNet50, DenseNet121, and ViT-B/16."""

import torch
import torch.nn as nn
import torchvision.models as models


class CNNBackbone(nn.Module):
    """
    Wrapper that loads a pretrained backbone and replaces the classifier
    head with a projection layer to output a fixed-dimension feature vector.
    """

    def __init__(
        self,
        name: str,
        output_dim: int = 512,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.name = name
        self.output_dim = output_dim
        self._has_spatial_target = True

        if name == "efficientnet":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            self._target_layer = self.backbone.features[-1]

        elif name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self._target_layer = self.backbone.layer4[-1]

        elif name == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            self.backbone = models.densenet121(weights=weights)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            self._target_layer = self.backbone.features[-1]

        elif name == "vit":
            weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
            self.backbone = models.vit_b_16(weights=weights)
            in_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Identity()
            self._target_layer = None
            self._has_spatial_target = False

        else:
            raise ValueError(f"Unknown backbone: {name}")

        self.projector = nn.Sequential(
            nn.Linear(in_features, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    @property
    def target_layer(self):
        return self._target_layer

    @property
    def has_spatial_target(self) -> bool:
        return self._has_spatial_target

    def get_freezable_params(self):
        if self.name == "resnet50":
            params = []
            for attr in ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4"):
                params.extend(getattr(self.backbone, attr).parameters())
            return params
        elif self.name == "vit":
            return list(self.backbone.parameters())
        elif hasattr(self.backbone, "features"):
            return list(self.backbone.features.parameters())
        return list(self.backbone.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.flatten(1)
        return self.projector(features)
