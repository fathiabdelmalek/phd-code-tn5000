from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import json


class BaseEvaluator(ABC):
    """Abstract base class for all evaluators.

    All evaluators (YOLO, Faster-RCNN, FCOS, etc.) must inherit from this class
    to ensure consistent evaluation interface across the pipeline.
    """

    def __init__(
        self,
        model,
        dataloader,
        exp_dir: str,
        class_names: Optional[List[str]] = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ):
        self.model = model
        self.dataloader = dataloader
        self.exp_dir = Path(exp_dir)
        self.class_names = class_names or ["class_0", "class_1"]
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / "results").mkdir(exist_ok=True)
        (self.exp_dir / "boxes").mkdir(exist_ok=True)
        (self.exp_dir / "heatmaps").mkdir(exist_ok=True)
        (self.exp_dir / "plots").mkdir(exist_ok=True)

        self._last_metrics: Dict[str, Any] = {}

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """Run full evaluation on test/val set.

        Returns:
            Dictionary of evaluation metrics.
        """
        pass

    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        """Run inference on a single image or batch."""
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Return the most recent evaluation metrics.

        Returns:
            Dictionary of metrics from last evaluation.
        """
        pass

    def save_results(
        self, metrics: Dict[str, Any], extra_data: Optional[Dict] = None
    ) -> None:
        """Save evaluation results to disk.

        Args:
            metrics: Dictionary of metrics to save.
            extra_data: Optional additional data to save (e.g., confusion matrix).
        """
        results_dir = self.exp_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        with open(results_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        if extra_data:
            if "confusion_matrix" in extra_data:
                self._save_confusion_matrix(extra_data["confusion_matrix"])
            if "pr_curves" in extra_data:
                self._save_pr_curves(extra_data["pr_curves"])
            if "train_loss" in extra_data and "val_loss" in extra_data:
                self._save_loss_curves(extra_data["train_loss"], extra_data["val_loss"])

        self._last_metrics = metrics
        print(f"Results saved to {results_dir}")

    def _save_confusion_matrix(self, matrix: Any) -> None:
        """Save confusion matrix visualization. Override in subclasses for custom styling."""
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        results_dir = self.exp_dir / "results"

        if isinstance(matrix, list):
            matrix = np.array(matrix)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".0f",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.savefig(results_dir / "confusion_matrix.jpg", dpi=150, bbox_inches="tight")
        plt.close()

    def _save_pr_curves(self, pr_data: Dict) -> None:
        """Save Precision-Recall curves. Override in subclasses for custom styling."""
        import matplotlib.pyplot as plt

        plots_dir = self.exp_dir / "plots"

        plt.figure(figsize=(10, 8))
        for cls_name, (recall, precision) in pr_data.items():
            plt.plot(recall, precision, label=cls_name, linewidth=2)

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / "pr_curves.jpg", dpi=150, bbox_inches="tight")
        plt.close()

    def _save_loss_curves(self, train_loss: List[float], val_loss: List[float]) -> None:
        """Save training/validation loss curves."""
        import matplotlib.pyplot as plt

        plots_dir = self.exp_dir / "plots"

        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label="Train Loss", linewidth=2)
        plt.plot(val_loss, label="Val Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / "loss_curves.jpg", dpi=150, bbox_inches="tight")
        plt.close()

    def generate_comparison_images(self) -> None:
        """Generate ground truth vs prediction comparison images.
        Override in subclasses for model-specific implementation."""
        pass

    def generate_heatmaps(self) -> None:
        """Generate activation heatmaps for feature visualization.
        Override in subclasses for model-specific implementation."""
        pass
