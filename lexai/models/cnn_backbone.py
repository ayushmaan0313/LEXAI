"""CNN backbone wrappers for EfficientNet, ResNet50, and DenseNet121."""

import torch
import torch.nn as nn
import torchvision.models as models


class CNNBackbone(nn.Module):
    """
    Wrapper that loads a pretrained CNN backbone and replaces the classifier
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

        if name == "efficientnet":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            # Store reference to the last conv layer for Grad-CAM
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

        else:
            raise ValueError(f"Unknown backbone: {name}")

        # Projection head: backbone features → output_dim
        self.projector = nn.Sequential(
            nn.Linear(in_features, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    @property
    def target_layer(self) -> nn.Module:
        """Return the target conv layer for Grad-CAM."""
        return self._target_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) input image tensor
        Returns:
            features: (B, output_dim)
        """
        features = self.backbone(x)
        # DenseNet returns (B, C, 1, 1) after adaptive pool in some versions
        if features.dim() > 2:
            features = features.flatten(1)
        return self.projector(features)
