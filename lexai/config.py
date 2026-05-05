"""Central configuration for the LEXAI project."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    image_size: Tuple[int, int] = (224, 224)
    num_classes: int = 4
    class_names: List[str] = field(
        default_factory=lambda: ["Normal", "ALL", "AML", "CML"]
    )
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    augmentation: bool = True
    use_stain_norm: bool = False
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class CNNConfig:
    """CNN ensemble configuration."""
    backbone_names: List[str] = field(
        default_factory=lambda: ["efficientnet", "resnet50", "densenet121", "vit"]
    )
    pretrained: bool = True
    global_feature_dim: int = 512
    dropout: float = 0.5
    use_vit: bool = True


@dataclass
class GNNConfig:
    """GNN spatial analysis configuration."""
    enabled: bool = True
    node_feature_dim: int = 64
    hidden_dim: int = 128
    spatial_feature_dim: int = 256
    num_gcn_layers: int = 2
    num_gat_layers: int = 2
    num_gat_heads: int = 4
    num_sage_layers: int = 1
    dropout: float = 0.4
    k_neighbors: int = 6
    min_cells: int = 3


@dataclass
class FusionConfig:
    """Multi-modal fusion configuration."""
    fused_dim: int = 512
    num_attention_heads: int = 8
    dropout: float = 0.4


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    learning_rate: float = 3e-4
    finetune_lr: float = 5e-5
    calibration_lr: float = 1e-2
    weight_decay: float = 1e-4
    epochs: int = 30
    patience: int = 15
    label_smoothing: float = 0.1
    classification_weight: float = 1.0
    spatial_score_weight: float = 0.3
    density_weight: float = 0.3
    mc_dropout_samples: int = 20
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    freeze_backbone_epochs: int = 10
    use_class_weights: bool = True
    use_weighted_sampler: bool = True


@dataclass
class ServerConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    model_checkpoint: str = "checkpoints/best_model.pth"
    calibrated_checkpoint: str = "checkpoints/calibrated_model.pth"
    device: str = "auto"


@dataclass
class LEXAIConfig:
    """Master configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


DEFAULT_CONFIG = LEXAIConfig()
