"""GNN Spatial Analysis Pathway.

Converts segmented cells into a graph and processes through
GCN → GAT → GraphSAGE to produce 256-dim spatial features.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import global_mean_pool, global_add_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

from lexai.config import GNNConfig, DEFAULT_CONFIG
from lexai.models.gnn_layers import GCNBlock, GATBlock, GraphSAGEBlock
from lexai.data.segmentation import CellInfo


class CellGraphBuilder:
    """
    Converts detected cells into a PyTorch Geometric Data object.

    Graph construction methods:
    - k-NN: Connect each cell to its k nearest spatial neighbors
    - Delaunay: Delaunay triangulation based on cell centroids
    """

    def __init__(self, k: int = 6, method: str = "knn"):
        self.k = k
        self.method = method

    def build_graph(
        self,
        cells: List[CellInfo],
        cell_features: np.ndarray,
    ) -> Optional["Data"]:
        """
        Build a PyG Data object from cell information.

        Args:
            cells: List of detected CellInfo
            cell_features: (N, feat_dim) array of per-cell features

        Returns:
            PyG Data object or None if not enough cells
        """
        if not HAS_PYG:
            raise ImportError("torch-geometric is required for graph building")

        n = len(cells)
        if n < 2:
            return None

        # Node features
        x = torch.tensor(cell_features, dtype=torch.float32)

        # Extract centroids
        centroids = np.array([c.centroid for c in cells], dtype=np.float64)

        # Build edges
        if self.method == "delaunay" and n >= 3:
            edge_index = self._delaunay_edges(centroids)
        else:
            edge_index = self._knn_edges(centroids)

        # Compute edge attributes (spatial distances)
        src, dst = edge_index[0], edge_index[1]
        distances = np.linalg.norm(
            centroids[src.numpy()] - centroids[dst.numpy()], axis=1
        )
        edge_attr = torch.tensor(distances, dtype=torch.float32).unsqueeze(-1)

        # Normalized centroid positions as additional features
        pos = torch.tensor(centroids, dtype=torch.float32)
        if pos.max() > 0:
            pos = pos / pos.max()  # Normalize to [0, 1]

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            num_nodes=n,
        )

        return data

    def _knn_edges(self, centroids: np.ndarray) -> torch.Tensor:
        """Build k-NN graph edges."""
        n = len(centroids)
        k = min(self.k, n - 1)

        dist_matrix = cdist(centroids, centroids)
        np.fill_diagonal(dist_matrix, np.inf)

        src_list, dst_list = [], []
        for i in range(n):
            neighbors = np.argsort(dist_matrix[i])[:k]
            for j in neighbors:
                src_list.append(i)
                dst_list.append(j)

        # Make bidirectional
        edge_index = torch.tensor(
            [src_list + dst_list, dst_list + src_list],
            dtype=torch.long
        )
        return edge_index

    def _delaunay_edges(self, centroids: np.ndarray) -> torch.Tensor:
        """Build Delaunay triangulation edges."""
        tri = Delaunay(centroids)
        src_list, dst_list = [], []

        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    src_list.extend([simplex[i], simplex[j]])
                    dst_list.extend([simplex[j], simplex[i]])

        edge_index = torch.tensor(
            [src_list, dst_list], dtype=torch.long
        )
        return edge_index


class GNNPathway(nn.Module):
    """
    Full GNN spatial analysis pathway.

    Architecture: GCN layers → GAT layers → GraphSAGE → global pooling
    Outputs a 256-dim spatial feature vector.
    """

    def __init__(self, config: GNNConfig = None):
        super().__init__()
        self.config = config or DEFAULT_CONFIG.gnn

        if not HAS_PYG:
            raise ImportError("torch-geometric is required for GNNPathway")

        self.graph_builder = CellGraphBuilder(
            k=self.config.k_neighbors, method="knn"
        )

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(self.config.node_feature_dim, self.config.hidden_dim),
            nn.BatchNorm1d(self.config.hidden_dim),
            nn.ReLU(inplace=True),
        )

        # GCN layers
        self.gcn_layers = nn.ModuleList()
        for i in range(self.config.num_gcn_layers):
            in_dim = self.config.hidden_dim
            self.gcn_layers.append(
                GCNBlock(in_dim, self.config.hidden_dim, self.config.dropout)
            )

        # GAT layers
        self.gat_layers = nn.ModuleList()
        gat_in = self.config.hidden_dim
        for i in range(self.config.num_gat_layers):
            heads = self.config.num_gat_heads
            gat_out = self.config.hidden_dim // heads  # So concat = hidden_dim
            is_last = (i == self.config.num_gat_layers - 1)
            self.gat_layers.append(
                GATBlock(
                    gat_in, gat_out, heads=heads,
                    dropout=self.config.dropout,
                    concat=True,
                )
            )
            gat_in = gat_out * heads

        # GraphSAGE layers
        self.sage_layers = nn.ModuleList()
        for i in range(self.config.num_sage_layers):
            self.sage_layers.append(
                GraphSAGEBlock(
                    self.config.hidden_dim,
                    self.config.hidden_dim,
                    self.config.dropout,
                )
            )

        # Output projection → spatial_feature_dim
        self.output_proj = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.spatial_feature_dim),
            nn.BatchNorm1d(self.config.spatial_feature_dim),
            nn.ReLU(inplace=True),
        )

        # Spatial pattern score predictor (from graph-level features)
        self.spatial_score_head = nn.Sequential(
            nn.Linear(self.config.spatial_feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        data: "Data",
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            data: PyG Data or Batch object
            return_attention: whether to return GAT attention weights

        Returns:
            dict with:
                'spatial_features': (B, 256) spatial feature vector
                'spatial_score': (B, 1) clustering metric
                'node_embeddings': (N, hidden) per-node embeddings
                'attention_weights': attention coefficients (if requested)
        """
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Input projection
        x = self.input_proj(x)

        # GCN layers with residual connections
        for gcn in self.gcn_layers:
            residual = x
            x = gcn(x, edge_index)
            x = x + residual  # Residual

        # GAT layers
        attn_weights = None
        for gat in self.gat_layers:
            residual = x
            if return_attention:
                x, attn_weights = gat(x, edge_index, return_attention=True)
            else:
                x = gat(x, edge_index)
            x = x + residual  # Residual

        # GraphSAGE layers
        for sage in self.sage_layers:
            residual = x
            x = sage(x, edge_index)
            x = x + residual  # Residual

        node_embeddings = x

        # Global pooling → graph-level representation
        graph_feat = global_mean_pool(x, batch)  # (B, hidden_dim)

        # Output projection
        spatial_features = self.output_proj(graph_feat)  # (B, 256)

        # Spatial pattern score
        spatial_score = self.spatial_score_head(spatial_features)

        result = {
            "spatial_features": spatial_features,
            "spatial_score": spatial_score,
            "node_embeddings": node_embeddings,
        }

        if return_attention and attn_weights is not None:
            result["attention_weights"] = attn_weights

        return result

    def build_graph_from_cells(
        self,
        cells: List[CellInfo],
        cell_features: np.ndarray,
        device: torch.device = None,
    ) -> Optional["Data"]:
        """Convenience method to build a graph and move to device."""
        data = self.graph_builder.build_graph(cells, cell_features)
        if data is not None and device is not None:
            data = data.to(device)
        return data

    def create_dummy_output(
        self, batch_size: int = 1, device: torch.device = None
    ) -> Dict[str, torch.Tensor]:
        """Create zero spatial features when no cells are detected."""
        device = device or torch.device("cpu")
        return {
            "spatial_features": torch.zeros(
                batch_size, self.config.spatial_feature_dim, device=device
            ),
            "spatial_score": torch.zeros(batch_size, 1, device=device),
            "node_embeddings": torch.zeros(1, self.config.hidden_dim, device=device),
        }
