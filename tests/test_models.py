import pytest
from abc import ABC
from unittest.mock import Mock, patch
import os

from src.models.common.cord_att import CoordAtt


class TestCoordAtt:
    """Tests for Coordinate Attention module."""

    def test_init(self):
        """Test CoordAtt initialization."""
        conv = CoordAtt(64, 64)
        assert conv.pool_h is not None
        assert conv.pool_w is not None

    def test_forward(self):
        """Test CoordAtt forward pass."""
        import torch

        conv = CoordAtt(64, 64)
        x = torch.randn(1, 64, 32, 32)
        out = conv(x)

        assert out.shape == x.shape

    def test_forward_different_sizes(self):
        """Test CoordAtt with different input sizes."""
        import torch

        conv = CoordAtt(128, 128)

        for size in [16, 32, 64]:
            x = torch.randn(2, 128, size, size)
            out = conv(x)
            assert out.shape == x.shape


class TestEntrypoints:
    """Tests for train.py and val.py entrypoints."""

    def test_train_entrypoint_imports(self):
        """Test that train.py imports correctly."""
        import sys
        import importlib.util

        spec = importlib.util.spec_from_file_location("train", "src/train.py")
        module = importlib.util.module_from_spec(spec)

        # This will fail if imports are broken
        spec.loader.exec_module(module)

    def test_val_entrypoint_imports(self):
        """Test that val.py imports correctly."""
        import sys
        import importlib.util

        spec = importlib.util.spec_from_file_location("val", "src/val.py")
        module = importlib.util.module_from_spec(spec)

        # This will fail if imports are broken
        spec.loader.exec_module(module)


class TestModelRegistry:
    """Additional tests for model registry."""

    def test_model_wrapper_interface(self):
        """Test that BaseModel interface is correct."""
        from src.models.base_model import BaseModel
        import torch.nn as nn

        assert issubclass(BaseModel, nn.Module)
        assert issubclass(BaseModel, ABC)

    def test_model_summary(self):
        """Test model summary method."""
        import torch.nn as nn
        from src.models.base_model import BaseModel

        class TestModel(BaseModel):
            def __init__(self):
                super().__init__(num_classes=2)
                self.conv = nn.Conv2d(3, 64, 3)

            def forward(self, x):
                return self.conv(x)

            def train_mode(self):
                pass

            def val_mode(self):
                pass

            def predict(self, x):
                return x

            def get_model(self):
                return self

            def load_weights(self, path):
                pass

            def save_weights(self, path):
                pass

        model = TestModel()
        summary = model.summary()

        assert summary["num_classes"] == 2
        assert summary["total_params"] > 0
        assert summary["trainable_params"] > 0


class TestExperimentMetadata:
    """Tests for experiment metadata saving."""

    def test_metadata_structure(self, temp_dir, mock_dataloader):
        """Test experiment metadata structure."""
        from src.core.base_trainer import BaseTrainer

        class ConcreteTrainer(BaseTrainer):
            def train(self):
                return {}

            def train_epoch(self):
                return {}

            def validate(self):
                return {}

            def save_checkpoint(self, f, o=True):
                pass

            def load_checkpoint(self, f):
                pass

        trainer = ConcreteTrainer(
            model=Mock(summary=lambda: {"params": 1000}),
            train_loader=mock_dataloader,
            val_loader=mock_dataloader,
            exp_dir=temp_dir,
            config={"epochs": 50, "lr": 0.001},
        )

        trainer._save_experiment_metadata()

        metadata_path = os.path.join(temp_dir, "experiment.json")
        assert os.path.exists(metadata_path)

        import json

        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "config" in metadata
        assert "current_epoch" in metadata
        assert "best_metric" in metadata
