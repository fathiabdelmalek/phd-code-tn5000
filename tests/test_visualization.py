import pytest
import os
import numpy as np
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_loss_curves,
    plot_pr_curves,
    plot_metrics_comparison,
    plot_metrics_by_epoch,
    save_visualization,
)


class TestPlotConfusionMatrix:
    """Tests for plot_confusion_matrix function."""

    def test_plot_basic(self, temp_dir):
        """Test basic confusion matrix plotting."""
        cm = np.array([[10, 2], [3, 15]])
        save_path = os.path.join(temp_dir, "cm.jpg")

        plot_confusion_matrix(cm, ["benign", "malignant"], save_path)

        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_plot_normalized(self, temp_dir):
        """Test normalized confusion matrix."""
        cm = np.array([[10, 2], [3, 15]])
        save_path = os.path.join(temp_dir, "cm_norm.jpg")

        plot_confusion_matrix(cm, ["a", "b"], save_path, normalize=True)

        assert os.path.exists(save_path)


class TestPlotLossCurves:
    """Tests for plot_loss_curves function."""

    def test_plot_basic(self, temp_dir):
        """Test basic loss curve plotting."""
        train_loss = [1.0, 0.8, 0.6, 0.4, 0.3]
        val_loss = [1.1, 0.9, 0.7, 0.5, 0.4]
        save_path = os.path.join(temp_dir, "loss.jpg")

        plot_loss_curves(train_loss, val_loss, save_path)

        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_plot_custom_labels(self, temp_dir):
        """Test with custom labels."""
        train_loss = [1.0, 0.5]
        val_loss = [1.2, 0.6]
        save_path = os.path.join(temp_dir, "loss_custom.jpg")

        plot_loss_curves(
            train_loss,
            val_loss,
            save_path,
            train_label="Training",
            val_label="Validation",
        )

        assert os.path.exists(save_path)


class TestPlotPRCurves:
    """Tests for plot_pr_curves function."""

    def test_plot_basic(self, temp_dir):
        """Test basic PR curve plotting."""
        recall = np.linspace(0, 1, 100)
        precision = 1 - recall * 0.5
        pr_data = {"benign": (recall, precision)}
        save_path = os.path.join(temp_dir, "pr.jpg")

        plot_pr_curves(pr_data, save_path)

        assert os.path.exists(save_path)

    def test_plot_multiple_classes(self, temp_dir):
        """Test PR curves for multiple classes."""
        recall = np.linspace(0, 1, 100)
        pr_data = {
            "benign": (recall, 1 - recall * 0.3),
            "malignant": (recall, 1 - recall * 0.5),
        }
        save_path = os.path.join(temp_dir, "pr_multi.jpg")

        plot_pr_curves(pr_data, save_path)

        assert os.path.exists(save_path)


class TestPlotMetricsComparison:
    """Tests for plot_metrics_comparison function."""

    def test_plot_basic(self, temp_dir):
        """Test basic metrics comparison."""
        metrics = {"map50": 0.85, "precision": 0.9, "recall": 0.8}
        save_path = os.path.join(temp_dir, "metrics.jpg")

        plot_metrics_comparison(metrics, save_path)

        assert os.path.exists(save_path)


class TestPlotMetricsByEpoch:
    """Tests for plot_metrics_by_epoch function."""

    def test_plot_basic(self, temp_dir):
        """Test basic metrics by epoch plotting."""
        history = {
            "loss": [1.0, 0.8, 0.6, 0.4],
            "accuracy": [0.5, 0.7, 0.8, 0.9],
        }
        save_path = os.path.join(temp_dir, "history.jpg")

        plot_metrics_by_epoch(history, save_path)

        assert os.path.exists(save_path)


class TestSaveVisualization:
    """Tests for the save_visualization factory function."""

    def test_save_confusion_matrix(self, temp_dir):
        """Test saving confusion matrix via factory."""
        cm = np.array([[10, 2], [3, 15]])
        save_path = os.path.join(temp_dir, "cm.jpg")

        save_visualization(save_path, "confusion_matrix", cm, class_names=["a", "b"])

        assert os.path.exists(save_path)

    def test_save_loss(self, temp_dir):
        """Test saving loss curves via factory."""
        data = {"train": [1.0, 0.5], "val": [1.1, 0.6]}
        save_path = os.path.join(temp_dir, "loss.jpg")

        save_visualization(save_path, "loss", data)

        assert os.path.exists(save_path)

    def test_unknown_plot_type(self, temp_dir):
        """Test that unknown plot type raises error."""
        with pytest.raises(ValueError):
            save_visualization(os.path.join(temp_dir, "test.jpg"), "unknown_type", {})
