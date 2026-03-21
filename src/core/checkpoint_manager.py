import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import json


class CheckpointManager:
    """Manages model checkpoint saving and loading."""

    def __init__(self, exp_dir: str):
        self.exp_dir = Path(exp_dir)
        self.weights_dir = self.exp_dir / "weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        self.best_path = self.weights_dir / "best.pt"
        self.last_path = self.weights_dir / "last.pt"

    def save(
        self,
        model,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        metrics: Optional[Dict] = None,
        is_best: bool = False,
        filepath: Optional[str] = None,
        include_optimizer: bool = True,
    ) -> str:
        """Save a checkpoint.

        Args:
            model: The model to save.
            optimizer: Optional optimizer state.
            scheduler: Optional scheduler state.
            epoch: Current epoch number.
            metrics: Dictionary of current metrics.
            is_best: Whether this is the best model so far.
            filepath: Custom filepath (overrides default).
            include_optimizer: Whether to include optimizer/scheduler state.

        Returns:
            Path to the saved checkpoint.
        """
        checkpoint = {
            "epoch": epoch,
            "metrics": metrics or {},
        }

        if hasattr(model, "state_dict"):
            checkpoint["model_state_dict"] = model.state_dict()
        else:
            checkpoint["model_state_dict"] = model

        if include_optimizer and optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None and hasattr(scheduler, "state_dict"):
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if filepath:
            save_path = filepath
        elif is_best:
            save_path = str(self.best_path)
        else:
            save_path = str(self.last_path)

        torch.save(checkpoint, save_path)
        return save_path

    def load(
        self,
        model,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        filepath: Optional[str] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Load a checkpoint.

        Args:
            model: The model to load weights into.
            optimizer: Optional optimizer to load state into.
            scheduler: Optional scheduler to load state into.
            filepath: Path to checkpoint file. If None, loads last.pt.
            device: Device to load checkpoint to.

        Returns:
            Dictionary containing checkpoint info (epoch, metrics, etc.).
        """
        if filepath is None:
            filepath = str(self.last_path)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=device)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return {
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
        }

    def get_best_weights(self) -> Optional[str]:
        """Return path to best weights if they exist."""
        return str(self.best_path) if self.best_path.exists() else None

    def get_last_weights(self) -> Optional[str]:
        """Return path to last weights if they exist."""
        return str(self.last_path) if self.last_path.exists() else None

    def save_metrics(self, metrics: Dict, name: str = "metrics.json") -> None:
        """Save metrics to a JSON file."""
        metrics_path = self.exp_dir / "results" / name
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
