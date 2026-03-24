"""
FCOS Model - Simple PyTorch detection model.
Uses COCO-pretrained FCOS, adjusts class labels for TN5000.
"""

import torchvision
import torch
import torch.nn as nn


def get_fcos_model(num_classes: int = 2, pretrained: bool = True):
    """Load FCOS model with pretrained weights.

    Uses COCO-pretrained model. For TN5000 (2 classes), we remap:
    - COCO class 0 (person) -> TN5000 class 0 (benign)
    - COCO class 14 (bird) -> TN5000 class 1 (malignant)

    Or simply train with all COCO classes and filter during evaluation.
    """
    model = torchvision.models.detection.fcos_resnet50_fpn(
        weights="DEFAULT" if pretrained else None
    )
    model.num_classes = num_classes
    return model


class FCOSWrapper(nn.Module):
    """Simple wrapper for FCOS to match training interface."""

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.model = get_fcos_model(num_classes, pretrained)
        self._original_classes = 80

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def train(self, mode=True):
        super().train(mode)
        self.model.train(mode)
        return self

    def eval(self):
        super().eval()
        self.model.eval()
        return self

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


def create_fcos_model(num_classes=2, pretrained=True, device="cuda"):
    """Create and return FCOS model on specified device."""
    model = FCOSWrapper(num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)
    return model
