"""Pydantic schemas for API request/response models."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class AnalysisResult(BaseModel):
    """Complete analysis result from LEXAI."""

    # Classification
    predicted_class: str = Field(
        ..., description="Predicted class: ALL, AML, CML, or Normal"
    )
    class_index: int = Field(..., description="Predicted class index")
    probabilities: Dict[str, float] = Field(
        ..., description="Class probabilities"
    )

    # Spatial Analysis
    spatial_pattern_score: float = Field(
        ..., description="Spatial clustering metric [0, 1]"
    )
    cell_count: int = Field(..., description="Number of detected cells")
    blast_percentage: float = Field(
        ..., description="Estimated blast cell percentage"
    )

    # Uncertainty
    confidence: float = Field(
        ..., description="Model confidence [0, 1]"
    )
    is_uncertain: bool = Field(
        ..., description="Whether prediction is flagged as uncertain"
    )
    prediction_variance: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-class prediction variance"
    )
    confidence_interval_low: Dict[str, float] = Field(
        default_factory=dict,
        description="5th percentile confidence interval"
    )
    confidence_interval_high: Dict[str, float] = Field(
        default_factory=dict,
        description="95th percentile confidence interval"
    )

    # Explainability (base64-encoded images)
    gradcam_heatmap: Optional[str] = Field(
        None, description="Base64-encoded Grad-CAM overlay image"
    )
    gnn_graph_visualization: Optional[str] = Field(
        None, description="Base64-encoded GNN graph overlay image"
    )

    # CNN ensemble details
    cnn_backbone_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Attention weights per CNN backbone"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    model_loaded: bool = False
    device: str = "cpu"
    version: str = "0.1.0"
