"""Unit tests for LEXAI model components."""

import sys
from pathlib import Path

import numpy as np
import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from lexai.config import LEXAIConfig, DEFAULT_CONFIG


@pytest.fixture
def config():
    return LEXAIConfig()


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def dummy_image():
    """Create a dummy batch of images: (B=2, 3, 224, 224)."""
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def single_image():
    """Single image: (1, 3, 224, 224)."""
    return torch.randn(1, 3, 224, 224)


class TestCNNBackbone:
    def test_efficientnet_output_shape(self, dummy_image):
        from lexai.models.cnn_backbone import CNNBackbone
        model = CNNBackbone("efficientnet", output_dim=512, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(dummy_image)
        assert out.shape == (2, 512), f"Expected (2, 512), got {out.shape}"

    def test_resnet50_output_shape(self, dummy_image):
        from lexai.models.cnn_backbone import CNNBackbone
        model = CNNBackbone("resnet50", output_dim=512, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(dummy_image)
        assert out.shape == (2, 512)

    def test_densenet121_output_shape(self, dummy_image):
        from lexai.models.cnn_backbone import CNNBackbone
        model = CNNBackbone("densenet121", output_dim=512, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(dummy_image)
        assert out.shape == (2, 512)

    def test_target_layer_exists(self):
        from lexai.models.cnn_backbone import CNNBackbone
        for name in ["efficientnet", "resnet50", "densenet121"]:
            model = CNNBackbone(name, pretrained=False)
            assert model.target_layer is not None


class TestCNNEnsemble:
    def test_ensemble_output_shape(self, dummy_image, config):
        from lexai.models.cnn_ensemble import CNNEnsemble
        config.cnn.pretrained = False
        model = CNNEnsemble(config.cnn)
        model.eval()
        num_backbones = len(model.backbone_names)
        with torch.no_grad():
            out = model(dummy_image)
        assert out["global_features"].shape == (2, 512)
        assert out["attention_weights"].shape == (2, num_backbones)
        assert len(out["backbone_features"]) == num_backbones

    def test_attention_weights_sum_to_one(self, dummy_image, config):
        from lexai.models.cnn_ensemble import CNNEnsemble
        config.cnn.pretrained = False
        model = CNNEnsemble(config.cnn)
        model.eval()
        with torch.no_grad():
            out = model(dummy_image)
        sums = out["attention_weights"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_ensemble_without_vit(self, dummy_image, config):
        from lexai.models.cnn_ensemble import CNNEnsemble
        config.cnn.pretrained = False
        config.cnn.use_vit = False
        model = CNNEnsemble(config.cnn)
        model.eval()
        assert len(model.backbone_names) == 3
        with torch.no_grad():
            out = model(dummy_image)
        assert out["global_features"].shape == (2, 512)


class TestCellSegmenter:
    def test_segmentation_on_synthetic(self):
        from lexai.data.segmentation import CellSegmenter
        # Create a synthetic image with some purple blobs
        img = np.ones((224, 224, 3), dtype=np.uint8) * 220  # light pink bg

        import cv2
        # Add purple circles (WBC-like)
        cv2.circle(img, (60, 60), 15, (100, 50, 150), -1)
        cv2.circle(img, (150, 100), 12, (80, 40, 140), -1)
        cv2.circle(img, (100, 180), 18, (90, 45, 160), -1)

        segmenter = CellSegmenter(min_cell_area=50)
        cells, mask = segmenter.segment(img)

        assert isinstance(cells, list)
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (224, 224)

    def test_feature_extraction(self):
        from lexai.data.segmentation import CellSegmenter, CellInfo
        segmenter = CellSegmenter()
        img = np.ones((224, 224, 3), dtype=np.uint8) * 200

        import cv2
        cv2.circle(img, (80, 80), 20, (100, 50, 150), -1)
        cv2.circle(img, (160, 160), 15, (80, 40, 140), -1)

        cells, _ = segmenter.segment(img)
        if len(cells) > 0:
            features = segmenter.extract_cell_features(img, cells, feature_dim=64)
            assert features.shape[1] == 64


class TestMultiModalFusion:
    def test_fusion_output_shape(self, config):
        from lexai.models.fusion import MultiModalFusion
        model = MultiModalFusion(config.cnn, config.gnn, config.fusion)
        model.eval()
        cnn_feat = torch.randn(2, 512)
        gnn_feat = torch.randn(2, 256)
        with torch.no_grad():
            out = model(cnn_feat, gnn_feat)
        assert out.shape == (2, config.fusion.fused_dim)


class TestLEXAIModel:
    def test_forward_without_graph(self, dummy_image, config):
        from lexai.models.lexai_model import LEXAIModel
        config.cnn.pretrained = False
        model = LEXAIModel(config)
        model.eval()
        with torch.no_grad():
            out = model(dummy_image, graph_data=None)
        assert out["logits"].shape == (2, config.data.num_classes)
        assert out["probabilities"].shape == (2, config.data.num_classes)
        assert out["predicted_class"].shape == (2,)
        assert out["blast_percentage"].shape == (2, 1)
        assert out["spatial_score"].shape == (2, 1)

    def test_class_name_mapping(self, config):
        from lexai.models.lexai_model import LEXAIModel
        config.cnn.pretrained = False
        model = LEXAIModel(config)
        assert model.get_class_name(0) == config.data.class_names[0]
        last_idx = config.data.num_classes - 1
        assert model.get_class_name(last_idx) == config.data.class_names[last_idx]


class TestMultiTaskLoss:
    def test_loss_computation(self, config):
        from lexai.training.losses import MultiTaskLoss
        criterion = MultiTaskLoss()
        predictions = {
            "logits": torch.randn(4, config.data.num_classes, requires_grad=True),
            "spatial_score": torch.rand(4, 1, requires_grad=True),
            "blast_percentage": torch.rand(4, 1, requires_grad=True) * 100,
        }
        targets = {
            "labels": torch.randint(0, config.data.num_classes, (4,)),
        }
        loss_dict = criterion(predictions, targets)
        assert "total_loss" in loss_dict
        assert loss_dict["total_loss"].requires_grad


class TestMetrics:
    def test_metrics_computation(self, config):
        from lexai.training.metrics import MetricsCalculator
        calc = MetricsCalculator(config.data.class_names)
        labels = np.array([0, 1, 2, 0, 0, 1])
        preds = np.array([0, 1, 2, 0, 1, 1])
        probs = np.zeros((6, config.data.num_classes))
        for i in range(6):
            probs[i, preds[i]] = 0.9
            for j in range(config.data.num_classes):
                if j != preds[i]:
                    probs[i, j] = 0.1 / 3
        calc.update(labels, preds, probs)
        metrics = calc.compute()
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1


class TestGradCAM:
    def test_gradcam_produces_heatmap(self, single_image, config):
        from lexai.models.cnn_backbone import CNNBackbone
        from lexai.explainability.gradcam import GradCAM

        config.cnn.pretrained = False
        backbone = CNNBackbone("resnet50", pretrained=False)
        cam = GradCAM(backbone, backbone.target_layer)

        heatmap = cam.generate(single_image, target_class=None)
        assert heatmap.shape == (224, 224)
        assert heatmap.min() >= 0
        assert heatmap.max() <= 1
        cam.release()


class TestUncertainty:
    def test_uncertainty_estimation(self, single_image, config):
        from lexai.models.lexai_model import LEXAIModel
        from lexai.uncertainty.estimator import UncertaintyEstimator

        config.cnn.pretrained = False
        model = LEXAIModel(config)
        estimator = UncertaintyEstimator(model, n_samples=5)
        result = estimator.estimate(single_image)

        assert "mean_probabilities" in result
        assert "confidence" in result
        assert result["confidence"].shape == (1,)
        assert result["mean_probabilities"].shape == (1, config.data.num_classes)
