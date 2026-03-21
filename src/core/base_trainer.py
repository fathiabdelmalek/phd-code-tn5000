from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Callable
import json


class BaseTrainer(ABC):
    """Abstract base class for all trainers.

    All trainers (YOLO, Faster-RCNN, FCOS, etc.) must inherit from this class
    to ensure consistent training interface across the pipeline.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        exp_dir: str,
        config: Optional[Dict] = None,
        callbacks: Optional[Dict[str, Callable]] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.exp_dir = Path(exp_dir)
        self.config = config or {}
        self.callbacks = callbacks or {}

        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / "weights").mkdir(exist_ok=True)
        (self.exp_dir / "logs").mkdir(exist_ok=True)

        self.current_epoch = 0
        self.best_metric = 0.0
        self.best_metric_name = "map50"
        self.patience = self.config.get("patience", 20)
        self.patience_counter = 0

        self.train_history = []
        self.val_history = []

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Main training loop. Must be implemented by subclasses.

        Returns:
            Final training metrics dictionary.
        """
        pass

    @abstractmethod
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch. Must be implemented by subclasses.

        Returns:
            Dictionary of training metrics for this epoch.
        """
        pass

    @abstractmethod
    def validate(self) -> Dict[str, float]:
        """Run validation. Must be implemented by subclasses.

        Returns:
            Dictionary of validation metrics.
        """
        pass

    @abstractmethod
    def save_checkpoint(self, filepath: str, include_optimizer: bool = True) -> None:
        """Save training checkpoint.

        Args:
            filepath: Path to save checkpoint.
            include_optimizer: Whether to include optimizer state.
        """
        pass

    @abstractmethod
    def load_checkpoint(self, filepath: str) -> None:
        """Load training checkpoint.

        Args:
            filepath: Path to checkpoint file.
        """
        pass

    def on_epoch_end(self, epoch: int, train_metrics: Dict, val_metrics: Dict) -> bool:
        """Hook called at the end of each epoch.

        Args:
            epoch: Current epoch number.
            train_metrics: Metrics from training.
            val_metrics: Metrics from validation.

        Returns:
            True if training should continue, False to stop early.
        """
        self.train_history.append(train_metrics)
        self.val_history.append(val_metrics)

        current_metric = val_metrics.get(self.best_metric_name, 0)
        is_best = current_metric > self.best_metric

        if is_best:
            self.best_metric = current_metric
            self.patience_counter = 0
            self.save_checkpoint(str(self.exp_dir / "weights" / "best.pt"))
        else:
            self.patience_counter += 1

        self.save_checkpoint(str(self.exp_dir / "weights" / "last.pt"))

        self._run_callback("on_epoch_end", epoch, train_metrics, val_metrics, is_best)

        if self.patience_counter >= self.patience:
            print(
                f"Early stopping triggered after {self.patience} epochs without improvement."
            )
            return False

        return True

    def _run_callback(self, name: str, *args, **kwargs) -> None:
        """Execute a callback if registered."""
        if name in self.callbacks:
            self.callbacks[name](*args, **kwargs)

    def register_callback(self, name: str, callback: Callable) -> None:
        """Register a callback function.

        Args:
            name: Callback name (e.g., 'on_epoch_end', 'on_train_start').
            callback: Callable to execute.
        """
        self.callbacks[name] = callback

    def get_history(self) -> Dict[str, list]:
        """Return training history."""
        return {
            "train": self.train_history,
            "val": self.val_history,
        }

    def _save_experiment_metadata(self) -> None:
        """Save experiment configuration and metadata."""
        metadata = {
            "config": self.config,
            "current_epoch": self.current_epoch,
            "best_metric": self.best_metric,
            "best_metric_name": self.best_metric_name,
            "patience": self.patience,
            "model_summary": self.model.summary()
            if hasattr(self.model, "summary")
            else {},
        }
        with open(self.exp_dir / "experiment.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
