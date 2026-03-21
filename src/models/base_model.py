from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Abstract base class for all model wrappers.

    All models (YOLO, Faster-RCNN, FCOS, etc.) must inherit from this class
    to ensure a consistent interface across the pipeline.
    """

    def __init__(self, num_classes: int, class_names: Optional[list] = None):
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self._is_training = False

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def train_mode(self) -> None:
        """Switch model to training mode with proper configuration."""
        pass

    @abstractmethod
    def val_mode(self) -> None:
        """Switch model to validation/inference mode with proper configuration."""
        pass

    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        """Run inference on input data."""
        pass

    @abstractmethod
    def get_model(self) -> nn.Module:
        """Return the underlying model instance."""
        pass

    @abstractmethod
    def load_weights(self, weights_path: str) -> None:
        """Load model weights from file."""
        pass

    @abstractmethod
    def save_weights(self, save_path: str) -> None:
        """Save model weights to file."""
        pass

    def get_device(self) -> torch.device:
        """Get the device the model is on."""
        return next(self.parameters()).device

    def summary(self) -> Dict[str, Any]:
        """Return model summary information."""
        return {
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }
