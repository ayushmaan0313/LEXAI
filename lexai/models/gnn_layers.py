"""GNN layers: GCN, GAT, and GraphSAGE blocks using PyTorch Geometric."""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class GCNBlock(nn.Module):
    """Graph Convolutional Network block with batch norm and dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        if not HAS_PYG:
            raise ImportError("torch-geometric is required for GNN layers")

        self.conv = GCNConv(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (N, in_channels) node features
            edge_index: (2, E) edge connectivity
        Returns:
            (N, out_channels) updated node features
        """
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class GATBlock(nn.Module):
    """Graph Attention Network block with multi-head attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.3,
        concat: bool = True,
    ):
        super().__init__()
        if not HAS_PYG:
            raise ImportError("torch-geometric is required for GNN layers")

        self.conv = GATConv(
            in_channels, out_channels, heads=heads,
            dropout=dropout, concat=concat
        )
        actual_out = out_channels * heads if concat else out_channels
        self.bn = nn.BatchNorm1d(actual_out)
        self.dropout = nn.Dropout(dropout)
        self.actual_out = actual_out

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Args:
            x: (N, in_channels) node features
            edge_index: (2, E) edge connectivity
            return_attention: if True, also return attention coefficients

        Returns:
            x: (N, out_channels * heads) updated node features
            attn_weights: (E, heads) attention coefficients (if requested)
        """
        if return_attention:
            x, (edge_idx, attn_weights) = self.conv(
                x, edge_index, return_attention_weights=True
            )
            x = self.bn(x)
            x = F.relu(x)
            x = self.dropout(x)
            return x, attn_weights
        else:
            x = self.conv(x, edge_index)
            x = self.bn(x)
            x = F.relu(x)
            x = self.dropout(x)
            return x


class GraphSAGEBlock(nn.Module):
    """GraphSAGE block for inductive graph learning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        if not HAS_PYG:
            raise ImportError("torch-geometric is required for GNN layers")

        self.conv = SAGEConv(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (N, in_channels) node features
            edge_index: (2, E) edge connectivity
        Returns:
            (N, out_channels) updated node features
        """
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x
