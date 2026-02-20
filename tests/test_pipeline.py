"""Integration tests for the LEXAI end-to-end pipeline."""

import sys
from pathlib import Path

import numpy as np
import torch
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from lexai.config import LEXAIConfig


@pytest.fixture
def config():
    cfg = LEXAIConfig()
    cfg.cnn.pretrained = False
    return cfg


@pytest.fixture
def dummy_pil_image():
    """Create a synthetic blood-smear-like PIL image."""
    arr = np.ones((224, 224, 3), dtype=np.uint8) * 220
    import cv2
    cv2.circle(arr, (60, 60), 15, (100, 50, 150), -1)
    cv2.circle(arr, (150, 100), 12, (80, 40, 140), -1)
    cv2.circle(arr, (100, 180), 18, (90, 45, 160), -1)
    # Add some red cells
    for x, y in [(30, 30), (100, 50), (170, 170), (50, 150)]:
        cv2.circle(arr, (x, y), 7, (180, 60, 60), -1)
    return Image.fromarray(arr[:, :, ::-1])  # BGR → RGB


class TestEndToEndPipeline:
    def test_full_pipeline_inference(self, config, dummy_pil_image):
        """Test complete inference from PIL image to results."""
        from api.inference import InferencePipeline

        pipeline = InferencePipeline(
            checkpoint_path=None,
            config=config,
            device="cpu",
        )

        result = pipeline.analyze(dummy_pil_image)

        # Check all required fields
        assert "predicted_class" in result
        assert result["predicted_class"] in config.data.class_names

        assert "probabilities" in result
        assert len(result["probabilities"]) == 4

        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1

        assert "cell_count" in result
        assert isinstance(result["cell_count"], int)

        assert "blast_percentage" in result
        assert "spatial_pattern_score" in result

        assert "is_uncertain" in result
        assert isinstance(result["is_uncertain"], bool)

        assert "cnn_backbone_weights" in result

    def test_preprocessing(self, config, dummy_pil_image):
        """Test image preprocessing pipeline."""
        from lexai.data.preprocessing import preprocess_image

        tensor = preprocess_image(dummy_pil_image, config.data)
        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == torch.float32

    def test_cell_segmentation_pipeline(self, dummy_pil_image):
        """Test cell segmentation on synthetic image."""
        import cv2
        from lexai.data.segmentation import CellSegmenter

        bgr = cv2.cvtColor(np.array(dummy_pil_image), cv2.COLOR_RGB2BGR)
        segmenter = CellSegmenter(min_cell_area=50)
        cells, mask = segmenter.segment(bgr)

        assert isinstance(cells, list)
        # Should detect at least some cells from our synthetic image
        # (exact number depends on thresholding)

        if len(cells) > 0:
            features = segmenter.extract_cell_features(bgr, cells, 64)
            assert features.shape[0] == len(cells)
            assert features.shape[1] == 64

    def test_model_process_single_image(self, config):
        """Test the model's single-image processing method."""
        from lexai.models.lexai_model import LEXAIModel
        from lexai.data.preprocessing import preprocess_image

        model = LEXAIModel(config)
        model.eval()

        # Create test image
        arr = np.ones((224, 224, 3), dtype=np.uint8) * 200
        import cv2
        cv2.circle(arr, (100, 100), 20, (100, 50, 150), -1)
        pil = Image.fromarray(arr[:, :, ::-1])

        tensor = preprocess_image(pil, config.data)
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        with torch.no_grad():
            result = model.process_single_image(tensor, bgr)

        assert "logits" in result
        assert "cells" in result
        assert "cell_count" in result
