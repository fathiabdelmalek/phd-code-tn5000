import pytest
from unittest.mock import Mock, MagicMock, patch


class TestPyTorchTrainer:
    """Tests for PyTorchTrainer."""

    def test_init(self, temp_dir, mock_model, mock_dataloader):
        """Test trainer initialization."""
        from src.core.trainers.pytorch_trainer import PyTorchTrainer

        trainer = PyTorchTrainer(
            model=mock_model,
            train_loader=mock_dataloader,
            val_loader=mock_dataloader,
            exp_dir=temp_dir,
        )

        assert trainer.device.type in ["cuda", "cpu"]
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_build_optimizer_adamw(self, temp_dir, mock_model, mock_dataloader):
        """Test building AdamW optimizer."""
        from src.core.trainers.pytorch_trainer import PyTorchTrainer

        trainer = PyTorchTrainer(
            model=mock_model,
            train_loader=mock_dataloader,
            val_loader=mock_dataloader,
            exp_dir=temp_dir,
            config={"optimizer": "AdamW"},
        )

        assert type(trainer.optimizer).__name__ == "AdamW"

    def test_build_optimizer_sgd(self, temp_dir, mock_model, mock_dataloader):
        """Test building SGD optimizer."""
        from src.core.trainers.pytorch_trainer import PyTorchTrainer

        trainer = PyTorchTrainer(
            model=mock_model,
            train_loader=mock_dataloader,
            val_loader=mock_dataloader,
            exp_dir=temp_dir,
            config={"optimizer": "SGD"},
        )

        assert type(trainer.optimizer).__name__ == "SGD"

    def test_build_scheduler_cosine(self, temp_dir, mock_model, mock_dataloader):
        """Test building cosine scheduler."""
        from src.core.trainers.pytorch_trainer import PyTorchTrainer

        trainer = PyTorchTrainer(
            model=mock_model,
            train_loader=mock_dataloader,
            val_loader=mock_dataloader,
            exp_dir=temp_dir,
            config={"cos_lr": True},
        )

        from torch.optim.lr_scheduler import CosineAnnealingLR

        assert isinstance(trainer.scheduler, CosineAnnealingLR)

    def test_build_scheduler_step(self, temp_dir, mock_model, mock_dataloader):
        """Test building step scheduler."""
        from src.core.trainers.pytorch_trainer import PyTorchTrainer

        trainer = PyTorchTrainer(
            model=mock_model,
            train_loader=mock_dataloader,
            val_loader=mock_dataloader,
            exp_dir=temp_dir,
            config={"cos_lr": False},
        )

        from torch.optim.lr_scheduler import StepLR

        assert isinstance(trainer.scheduler, StepLR)

    def test_unknown_optimizer_raises_error(
        self, temp_dir, mock_model, mock_dataloader
    ):
        """Test that unknown optimizer raises error."""
        from src.core.trainers.pytorch_trainer import PyTorchTrainer

        with pytest.raises(ValueError, match="Unknown optimizer"):
            PyTorchTrainer(
                model=mock_model,
                train_loader=mock_dataloader,
                val_loader=mock_dataloader,
                exp_dir=temp_dir,
                config={"optimizer": "UnknownOpt"},
            )


class TestPyTorchEvaluator:
    """Tests for PyTorchEvaluator."""

    def test_init(self, temp_dir, mock_model, mock_dataloader):
        """Test evaluator initialization."""
        from src.core.evaluators.pytorch_evaluator import PyTorchEvaluator

        evaluator = PyTorchEvaluator(
            model=mock_model,
            dataloader=mock_dataloader,
            exp_dir=temp_dir,
        )

        assert evaluator.device.type in ["cuda", "cpu"]
        assert evaluator.num_classes == 3

    def test_compute_metrics(self, temp_dir, mock_model, mock_dataloader):
        """Test computing metrics."""
        from src.core.evaluators.pytorch_evaluator import PyTorchEvaluator

        evaluator = PyTorchEvaluator(
            model=mock_model,
            dataloader=mock_dataloader,
            exp_dir=temp_dir,
        )

        def make_mock_tensor(arr):
            m = MagicMock()
            m.cpu.return_value = m
            m.numpy.return_value = arr
            return m

        predictions = [
            {"labels": make_mock_tensor([1, 2]), "boxes": make_mock_tensor([])},
            {"labels": make_mock_tensor([1]), "boxes": make_mock_tensor([])},
        ]
        targets = [
            {"labels": make_mock_tensor([1, 2]), "boxes": make_mock_tensor([])},
            {"labels": make_mock_tensor([1]), "boxes": make_mock_tensor([])},
        ]

        metrics = evaluator._compute_metrics(predictions, targets)

        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

    def test_compute_confusion_matrix(self, temp_dir, mock_model, mock_dataloader):
        """Test computing confusion matrix."""
        from src.core.evaluators.pytorch_evaluator import PyTorchEvaluator

        evaluator = PyTorchEvaluator(
            model=mock_model,
            dataloader=mock_dataloader,
            exp_dir=temp_dir,
        )

        def make_mock_tensor(arr):
            m = MagicMock()
            m.cpu.return_value = m
            m.numpy.return_value = arr
            return m

        predictions = [
            {"labels": make_mock_tensor([1]), "boxes": make_mock_tensor([])},
            {"labels": make_mock_tensor([0]), "boxes": make_mock_tensor([])},
        ]
        targets = [
            {"labels": make_mock_tensor([1]), "boxes": make_mock_tensor([])},
            {"labels": make_mock_tensor([1]), "boxes": make_mock_tensor([])},
        ]

        cm = evaluator._compute_confusion_matrix(predictions, targets)

        assert isinstance(cm, list)
        assert len(cm) == 3
