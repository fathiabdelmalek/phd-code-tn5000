import os
import pytest
from abc import ABC
from pathlib import Path
from src.core.base_trainer import BaseTrainer
from src.core.base_evaluator import BaseEvaluator
from src.datasets.base_dataset import BaseDataset
from src.models.base_model import BaseModel


class ConcreteTrainer(BaseTrainer):
    """Concrete implementation for testing."""

    def train(self):
        return {}

    def train_epoch(self):
        return {"loss": 0.5}

    def validate(self):
        return {"map50": 0.8}

    def save_checkpoint(self, filepath, include_optimizer=True):
        pass

    def load_checkpoint(self, filepath):
        pass


class ConcreteEvaluator(BaseEvaluator):
    """Concrete implementation for testing."""

    def evaluate(self):
        return {"map50": 0.85}

    def predict(self, *args, **kwargs):
        return {}

    def get_metrics(self):
        return {"map50": 0.85}


class ConcreteDataset(BaseDataset):
    """Concrete implementation for testing."""

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        import torch

        return torch.zeros(3, 224, 224), {"boxes": torch.zeros(0, 4)}


class ConcreteModel(BaseModel):
    """Concrete implementation for testing."""

    def __init__(self):
        super().__init__(num_classes=2)

    def forward(self, *args, **kwargs):
        return args

    def train_mode(self):
        pass

    def val_mode(self):
        pass

    def predict(self, *args, **kwargs):
        return {}

    def get_model(self):
        return self

    def load_weights(self, weights_path):
        pass

    def save_weights(self, save_path):
        pass


class TestBaseTrainer:
    """Tests for BaseTrainer abstract class."""

    def test_init(self, temp_dir, mock_dataloader):
        """Test trainer initialization."""
        trainer = ConcreteTrainer(
            model=None,
            train_loader=mock_dataloader,
            val_loader=mock_dataloader,
            exp_dir=temp_dir,
        )

        assert trainer.current_epoch == 0
        assert trainer.best_metric == 0.0
        assert trainer.patience == 20
        assert os.path.exists(os.path.join(temp_dir, "weights"))

    def test_on_epoch_end_improves(self, temp_dir, mock_dataloader):
        """Test on_epoch_end when metric improves."""
        trainer = ConcreteTrainer(
            model=None,
            train_loader=mock_dataloader,
            val_loader=mock_dataloader,
            exp_dir=temp_dir,
        )

        trainer.model = type("MockModel", (), {"summary": lambda: {}})()

        should_continue = trainer.on_epoch_end(0, {"loss": 0.5}, {"map50": 0.9})

        assert should_continue is True
        assert trainer.best_metric == 0.9
        assert trainer.patience_counter == 0

    def test_on_epoch_end_no_improvement(self, temp_dir, mock_dataloader):
        """Test on_epoch_end when metric doesn't improve."""
        trainer = ConcreteTrainer(
            model=None,
            train_loader=mock_dataloader,
            val_loader=mock_dataloader,
            exp_dir=temp_dir,
        )
        trainer.best_metric = 0.9

        trainer.model = type("MockModel", (), {"summary": lambda: {}})()

        should_continue = trainer.on_epoch_end(0, {"loss": 0.5}, {"map50": 0.8})

        assert should_continue is True
        assert trainer.patience_counter == 1

    def test_callbacks(self, temp_dir, mock_dataloader):
        """Test callback registration and execution."""
        trainer = ConcreteTrainer(
            model=None,
            train_loader=mock_dataloader,
            val_loader=mock_dataloader,
            exp_dir=temp_dir,
        )

        callback_called = []
        trainer.register_callback(
            "on_epoch_end", lambda *args: callback_called.append(True)
        )

        trainer.model = type("MockModel", (), {"summary": lambda: {}})()
        trainer.on_epoch_end(0, {}, {"map50": 0.5})

        assert len(callback_called) == 1

    def test_get_history(self, temp_dir, mock_dataloader):
        """Test getting training history."""
        trainer = ConcreteTrainer(
            model=None,
            train_loader=mock_dataloader,
            val_loader=mock_dataloader,
            exp_dir=temp_dir,
        )

        trainer.model = type("MockModel", (), {"summary": lambda: {}})()
        trainer.on_epoch_end(0, {"loss": 0.5}, {"map50": 0.8})

        history = trainer.get_history()
        assert "train" in history
        assert "val" in history
        assert len(history["train"]) == 1


class TestBaseEvaluator:
    """Tests for BaseEvaluator abstract class."""

    def test_init(self, temp_dir):
        """Test evaluator initialization."""
        evaluator = ConcreteEvaluator(
            model=None,
            dataloader=None,
            exp_dir=temp_dir,
        )

        assert str(evaluator.exp_dir) == str(temp_dir)
        assert evaluator.class_names == ["class_0", "class_1"]
        assert evaluator.conf_thres == 0.25
        assert evaluator.iou_thres == 0.45

    def test_save_results(self, temp_dir):
        """Test saving evaluation results."""
        evaluator = ConcreteEvaluator(
            model=None,
            dataloader=None,
            exp_dir=temp_dir,
        )

        metrics = {"map50": 0.85, "precision": 0.9}
        evaluator.save_results(metrics)

        results_path = os.path.join(str(temp_dir), "results", "metrics.json")
        assert os.path.exists(results_path)


class TestBaseDataset:
    """Tests for BaseDataset abstract class."""

    def test_get_class_names(self):
        """Test getting class names."""
        dataset = ConcreteDataset(".", class_names=["a", "b"])
        assert dataset.get_class_names() == ["a", "b"]

    def test_get_num_classes(self):
        """Test getting number of classes."""
        dataset = ConcreteDataset(".", class_names=["a", "b", "c"])
        assert dataset.get_num_classes() == 3

    def test_get_split(self):
        """Test getting current split."""
        dataset = ConcreteDataset(".", split="test")
        assert dataset.get_split() == "test"


class TestBaseModel:
    """Tests for BaseModel abstract class."""

    def test_init(self):
        """Test model initialization."""
        model = ConcreteModel()
        assert model.num_classes == 2
        assert model.class_names == ["class_0", "class_1"]

    def test_summary(self):
        """Test model summary."""
        model = ConcreteModel()
        summary = model.summary()

        assert "num_classes" in summary
        assert "class_names" in summary
        assert "total_params" in summary
        assert "trainable_params" in summary
