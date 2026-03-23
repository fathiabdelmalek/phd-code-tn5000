import pytest
from unittest.mock import Mock, patch


class TestTrainerRegistry:
    """Tests for trainer registry."""

    def test_get_trainer_yolo(self):
        """Test getting YOLO trainer."""
        from src.core.trainers.registry import get_trainer

        with patch("src.core.trainers.yolo_trainer.YOLOTrainer") as mock_yolo:
            mock_instance = Mock()
            mock_yolo.return_value = mock_instance

            trainer = get_trainer(
                model_name="yolo26n_ca",
                data_root="data/",
                exp_dir="experiments/test",
            )

            mock_yolo.assert_called_once()

    def test_get_trainer_pytorch(self):
        """Test getting PyTorch trainer for Faster-RCNN."""
        from src.core.trainers.registry import get_trainer

        with patch("src.core.trainers.pytorch_trainer.PyTorchTrainer") as mock_pytorch:
            with patch("src.models.registery.get_model") as mock_get_model:
                mock_model = Mock()
                mock_get_model.return_value = mock_model
                mock_instance = Mock()
                mock_pytorch.return_value = mock_instance

                trainer = get_trainer(
                    model_name="faster_rcnn",
                    train_loader=Mock(),
                    val_loader=Mock(),
                    exp_dir="experiments/test",
                )

                mock_pytorch.assert_called_once()

    def test_get_trainer_unknown(self):
        """Test that unknown model type raises error."""
        from src.core.trainers.registry import get_trainer

        with pytest.raises(ValueError, match="Unknown model type"):
            get_trainer(
                model_name="unknown_model",
                exp_dir="experiments/test",
            )


class TestEvaluatorRegistry:
    """Tests for evaluator registry."""

    def test_get_evaluator_yolo(self):
        """Test getting YOLO evaluator."""
        from src.core.evaluators.registry import get_evaluator

        with patch("src.core.evaluators.yolo_evaluator.YOLOEvaluator") as mock_yolo:
            mock_instance = Mock()
            mock_yolo.return_value = mock_instance

            evaluator = get_evaluator(
                model_name="yolo26n_ca",
                model=Mock(),
                data_cfg="data/test.yaml",
                exp_dir="experiments/test",
            )

            mock_yolo.assert_called_once()

    def test_get_evaluator_pytorch(self):
        """Test getting PyTorch evaluator."""
        from src.core.evaluators.registry import get_evaluator

        with patch(
            "src.core.evaluators.pytorch_evaluator.PyTorchEvaluator"
        ) as mock_pytorch:
            mock_instance = Mock()
            mock_pytorch.return_value = mock_instance

            evaluator = get_evaluator(
                model_name="faster_rcnn",
                model=Mock(),
                dataloader=Mock(),
                exp_dir="experiments/test",
            )

            mock_pytorch.assert_called_once()

    def test_get_evaluator_unknown(self):
        """Test that unknown model type raises error."""
        from src.core.evaluators.registry import get_evaluator

        with pytest.raises(ValueError, match="Unknown model type"):
            get_evaluator(
                model_name="unknown_model",
                model=Mock(),
                exp_dir="experiments/test",
            )


class TestModelRegistry:
    """Tests for model registry."""

    def test_get_model_yolo(self):
        """Test getting YOLO model."""
        from src.models.registery import get_model

        with patch("src.models.registery.YOLO") as mock_yolo:
            mock_instance = Mock()
            mock_instance.model = Mock()
            mock_yolo.return_value = mock_instance

            model = get_model("yolo26n_ca")

            mock_yolo.assert_called_once()

    def test_get_model_fcos(self):
        """Test getting FCOS model via the pytorch registry."""
        from src.models.pytorch import MODEL_REGISTRY

        mock_class = Mock()
        MODEL_REGISTRY["test_fcos"] = mock_class

        from src.models.registery import get_model

        model = get_model("test_fcos", num_classes=2)

        mock_class.assert_called_once_with(num_classes=2)

    def test_get_model_unknown(self):
        """Test that unknown model raises error."""
        from src.models.registery import get_model

        with pytest.raises(ValueError, match="Unknown model"):
            get_model("unknown_model")

    def test_list_available_models(self):
        """Test listing available models."""
        from src.models import list_available_models

        models = list_available_models()

        assert "yolo" in models
        assert "pytorch" in models
        assert "all" in models
        assert "fcos" in models["pytorch"]
        assert "yolo26n_ca" in models["yolo"]
