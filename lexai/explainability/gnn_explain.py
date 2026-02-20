"""GNN Attention Visualization for spatial explainability."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from lexai.data.segmentation import CellInfo


class GNNExplainer:
    """
    Visualizes GNN attention weights to explain which cell-cell
    relationships were most important for the prediction.
    """

    @staticmethod
    def extract_attention_weights(
        gnn_output: Dict[str, torch.Tensor],
    ) -> Optional[np.ndarray]:
        """
        Extract GAT attention coefficients from GNN output.

        Args:
            gnn_output: dict from GNNPathway forward pass

        Returns:
            attention_weights: (E, heads) numpy array or None
        """
        attn = gnn_output.get("gnn_attention_weights", None)
        if attn is None:
            attn = gnn_output.get("attention_weights", None)
        if attn is not None:
            return attn.detach().cpu().numpy()
        return None

    @staticmethod
    def visualize_cell_graph(
        image: np.ndarray,
        cells: List[CellInfo],
        edge_index: Optional[torch.Tensor] = None,
        attention_weights: Optional[np.ndarray] = None,
        node_importance: Optional[np.ndarray] = None,
        alpha: float = 0.7,
    ) -> np.ndarray:
        """
        Draw the cell graph overlaid on the original image.

        - Nodes (cells) shown as circles, colored by importance
        - Edges shown as lines, thickness proportional to attention weight

        Args:
            image: (H, W, 3) BGR uint8 image
            cells: List of CellInfo
            edge_index: (2, E) edge connectivity
            attention_weights: (E,) or (E, heads) attention values
            node_importance: (N,) per-node importance scores
            alpha: overlay transparency

        Returns:
            visualization: (H, W, 3) BGR uint8 image
        """
        vis = image.copy()

        if not cells:
            return vis

        # Process attention weights
        if attention_weights is not None and attention_weights.ndim > 1:
            # Average across heads
            attention_weights = attention_weights.mean(axis=-1)

        # Draw edges
        if edge_index is not None:
            edge_index_np = edge_index.cpu().numpy() if isinstance(
                edge_index, torch.Tensor
            ) else edge_index

            num_edges = edge_index_np.shape[1]

            for e in range(num_edges):
                src_idx = edge_index_np[0, e]
                dst_idx = edge_index_np[1, e]

                if src_idx >= len(cells) or dst_idx >= len(cells):
                    continue

                src_pt = cells[src_idx].centroid
                dst_pt = cells[dst_idx].centroid

                # Edge weight determines thickness and color
                if attention_weights is not None and e < len(attention_weights):
                    weight = float(attention_weights[e])
                    weight = min(max(weight, 0), 1)  # Clamp to [0, 1]
                    thickness = max(1, int(weight * 6))
                    # Color: low → blue, high → red
                    color = (
                        int(255 * (1 - weight)),
                        int(100 * (1 - weight)),
                        int(255 * weight),
                    )
                else:
                    thickness = 1
                    color = (200, 200, 200)

                cv2.line(vis, src_pt, dst_pt, color, thickness, cv2.LINE_AA)

        # Draw nodes
        for i, cell in enumerate(cells):
            cx, cy = cell.centroid
            radius = max(5, int(np.sqrt(cell.area / np.pi) * 0.3))

            if node_importance is not None and i < len(node_importance):
                importance = float(node_importance[i])
                importance = min(max(importance, 0), 1)
                # Green → Red gradient
                color = (
                    0,
                    int(255 * (1 - importance)),
                    int(255 * importance),
                )
            else:
                color = (0, 255, 0)  # Default green

            cv2.circle(vis, (cx, cy), radius, color, 2, cv2.LINE_AA)
            cv2.circle(vis, (cx, cy), 3, (255, 255, 255), -1)

        # Blend with original
        result = cv2.addWeighted(image, 1 - alpha, vis, alpha, 0)

        return result

    @staticmethod
    def compute_node_importance(
        node_embeddings: torch.Tensor,
        method: str = "norm",
    ) -> np.ndarray:
        """
        Compute per-node importance scores from node embeddings.

        Methods:
            'norm': L2 norm of each node's embedding
            'variance': variance across embedding dimensions

        Args:
            node_embeddings: (N, D) tensor
            method: scoring method

        Returns:
            importance: (N,) normalized importance scores
        """
        embeddings = node_embeddings.detach().cpu().numpy()

        if method == "norm":
            scores = np.linalg.norm(embeddings, axis=1)
        elif method == "variance":
            scores = np.var(embeddings, axis=1)
        else:
            scores = np.ones(len(embeddings))

        # Normalize to [0, 1]
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min > 1e-8:
            scores = (scores - s_min) / (s_max - s_min)

        return scores
