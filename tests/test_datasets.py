import pytest
import os
from unittest.mock import Mock, patch
from src.datasets.base_dataset import BaseDataset
from src.datasets.factory import get_dataloader


class DummyDataset(BaseDataset):
    """Concrete dataset for testing."""

    def __len__(self):
        return 5

    def __getitem__(self, idx):
        import torch

        return torch.zeros(3, 224, 224), {"boxes": torch.zeros(1, 4)}


class TestBaseDataset:
    """Tests for BaseDataset."""

    def test_dataset_creation(self):
        """Test creating a dataset."""
        dataset = DummyDataset("data/", split="train", class_names=["a", "b"])

        assert len(dataset) == 5
        assert dataset.get_split() == "train"
        assert dataset.get_class_names() == ["a", "b"]
        assert dataset.get_num_classes() == 2

    def test_getitem(self):
        """Test getting item from dataset."""
        dataset = DummyDataset("data/")
        img, target = dataset[0]

        assert img.shape == (3, 224, 224)
        assert "boxes" in target


class TestDataLoaderFactory:
    """Tests for dataloader factory."""

    def test_get_dataloader_yolo(self):
        """Test getting YOLO dataloader returns data config path."""
        with patch("src.datasets.yolo_loader.os.path.exists") as mock_exists:
            mock_exists.return_value = True

            with patch("src.datasets.yolo_loader.yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {"path": "data/"}

                result = get_dataloader("yolo26n_ca", "data/", split="train")

                assert result is not None
                assert isinstance(result, str)

    def test_get_dataloader_voc(self):
        """Test getting VOC dataloader returns DataLoader."""
        with patch("src.datasets.factory.ThyroidVOCDataset") as mock_dataset:
            mock_instance = Mock()
            mock_instance.__len__ = Mock(return_value=100)
            mock_dataset.return_value = mock_instance

            result = get_dataloader("faster_rcnn", "data/", split="train")

            mock_dataset.assert_called_once()
            from torch.utils.data import DataLoader

            assert isinstance(result, DataLoader)

    def test_get_dataloader_unknown_uses_voc(self):
        """Test that unknown model type falls back to VOC loader."""
        with patch("src.datasets.factory.ThyroidVOCDataset") as mock_dataset:
            mock_instance = Mock()
            mock_instance.__len__ = Mock(return_value=100)
            mock_dataset.return_value = mock_instance

            result = get_dataloader("unknown", "data/", split="train")

            mock_dataset.assert_called_once()
