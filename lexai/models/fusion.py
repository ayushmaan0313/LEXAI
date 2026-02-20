"""Multi-Modal Fusion: combines CNN global features and GNN spatial features."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lexai.config import FusionConfig, CNNConfig, GNNConfig, DEFAULT_CONFIG


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism.

    Allows CNN and GNN features to attend to each other,
    learning which modality is more informative for each input.
    """

    def __init__(
        self,
        cnn_dim: int = 512,
        gnn_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.total_dim = cnn_dim + gnn_dim

        # Project both modalities to same space
        self.cnn_proj = nn.Linear(cnn_dim, self.total_dim)
        self.gnn_proj = nn.Linear(gnn_dim, self.total_dim)

        # Multi-head self-attention on concatenated features
        self.attention = nn.MultiheadAttention(
            embed_dim=self.total_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(self.total_dim)

    def forward(
        self, cnn_feat: torch.Tensor, gnn_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            cnn_feat: (B, cnn_dim)
            gnn_feat: (B, gnn_dim)

        Returns:
            fused: (B, total_dim)
        """
        # Project to common space
        cnn_proj = self.cnn_proj(cnn_feat)  # (B, total_dim)
        gnn_proj = self.gnn_proj(gnn_feat)  # (B, total_dim)

        # Stack as sequence of 2 tokens: (B, 2, total_dim)
        tokens = torch.stack([cnn_proj, gnn_proj], dim=1)

        # Self-attention
        attn_out, _ = self.attention(tokens, tokens, tokens)

        # Combine attended tokens
        fused = attn_out.mean(dim=1)  # (B, total_dim)
        fused = self.norm(fused)

        return fused


class MultiModalFusion(nn.Module):
    """
    Multi-modal fusion module combining CNN and GNN pathways.

    Pipeline:
    1. Concatenate CNN (512) + GNN (256) = 768-dim
    2. Cross-modal attention to learn inter-modal interactions
    3. Final projection to fused_dim (512)
    """

    def __init__(
        self,
        cnn_config: CNNConfig = None,
        gnn_config: GNNConfig = None,
        fusion_config: FusionConfig = None,
    ):
        super().__init__()
        cnn_config = cnn_config or DEFAULT_CONFIG.cnn
        gnn_config = gnn_config or DEFAULT_CONFIG.gnn
        fusion_config = fusion_config or DEFAULT_CONFIG.fusion

        cnn_dim = cnn_config.global_feature_dim  # 512
        gnn_dim = gnn_config.spatial_feature_dim  # 256
        total_dim = cnn_dim + gnn_dim               # 768

        # Cross-modal attention
        self.cross_attention = CrossModalAttention(
            cnn_dim=cnn_dim,
            gnn_dim=gnn_dim,
            num_heads=fusion_config.num_attention_heads,
            dropout=fusion_config.dropout,
        )

        # Gating mechanism — learn how much each modality contributes
        self.gate = nn.Sequential(
            nn.Linear(total_dim, 2),
            nn.Softmax(dim=-1),
        )

        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(total_dim, fusion_config.fused_dim),
            nn.BatchNorm1d(fusion_config.fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(fusion_config.dropout),
            nn.Linear(fusion_config.fused_dim, fusion_config.fused_dim),
            nn.BatchNorm1d(fusion_config.fused_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        cnn_features: torch.Tensor,
        gnn_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cnn_features: (B, 512) global CNN features
            gnn_features: (B, 256) spatial GNN features

        Returns:
            fused: (B, fused_dim) fused representation
        """
        # Cross-modal attention
        fused = self.cross_attention(cnn_features, gnn_features)

        # Final projection
        output = self.output_proj(fused)

        return output
