"""Central configuration for the LEXAI project."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    image_size: Tuple[int, int] = (224, 224)
    num_classes: int = 5
    class_names: List[str] = field(
        default_factory=lambda: [
            "ALL_Blast", "ALL_Early_Pre_B", "ALL_Pre_B", "ALL_Pro_B", "Benign"
        ]
    )
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    augmentation: bool = True
    # ImageNet normalization stats
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class CNNConfig:
    """CNN ensemble configuration."""
    backbone_names: List[str] = field(
        default_factory=lambda: ["efficientnet", "resnet50", "densenet121"]
    )
    pretrained: bool = True
    global_feature_dim: int = 512
    dropout: float = 0.5


@dataclass
class GNNConfig:
    """GNN spatial analysis configuration."""
    node_feature_dim: int = 64
    hidden_dim: int = 128
    spatial_feature_dim: int = 256
    num_gcn_layers: int = 2
    num_gat_layers: int = 2
    num_gat_heads: int = 4
    num_sage_layers: int = 1
    dropout: float = 0.4
    k_neighbors: int = 6  # For k-NN graph construction
    min_cells: int = 3    # Minimum cells to build graph


@dataclass
class FusionConfig:
    """Multi-modal fusion configuration."""
    fused_dim: int = 512
    num_attention_heads: int = 8
    dropout: float = 0.4


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    epochs: int = 50
    patience: int = 15  # Early stopping
    # Multi-task loss weights
    classification_weight: float = 1.0
    spatial_score_weight: float = 0.3
    density_weight: float = 0.3
    # Uncertainty estimation
    mc_dropout_samples: int = 20
    # Scheduler
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    # Backbone freezing: freeze pretrained CNN backbones for N epochs
    freeze_backbone_epochs: int = 5
    # Class weighting: auto-compute inverse frequency weights
    use_class_weights: bool = True


@dataclass
class ServerConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    model_checkpoint: str = "checkpoints/best_model.pth"
    device: str = "auto"  # auto, cpu, or cuda


@dataclass
class LEXAIConfig:
    """Master configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


# Global default config
DEFAULT_CONFIG = LEXAIConfig()
