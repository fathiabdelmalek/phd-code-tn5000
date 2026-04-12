"""
FCOS Model - PyTorch detection model with 2-class output for TN5000.
Modifies the classification head to output TN5000 classes directly.
"""

import torchvision
import torch.nn as nn
import math


def get_fcos_model(num_classes: int = 2, pretrained: bool = True, use_coord_att: bool = False):
    """
    Load FCOS model modified for TN5000 (2 classes).

    Instead of using COCO pretrained weights with label remapping,
    we modify the head to output 2 classes directly.
    """
    model = torchvision.models.detection.fcos_resnet50_fpn(
        weights="DEFAULT" if pretrained else None
    )

    new_cls_logits = nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=1)

    # Initialize weights using Xavier and bias for low prior probability
    nn.init.normal_(new_cls_logits.weight, std=0.01)
    prior_prob = 0.01
    nn.init.constant_(new_cls_logits.bias, -math.log((1 - prior_prob) / prior_prob))

    model.head.classification_head.cls_logits = new_cls_logits
    model.head.classification_head.num_classes = num_classes

    return model


class FCOSWrapper(nn.Module):
    """Simple wrapper for FCOS to match training interface."""

    def __init__(self, num_classes=2, pretrained=True, use_coord_att=False):
        super().__init__()
        self.num_classes = num_classes
        self.use_coord_att = use_coord_att
        self.model = get_fcos_model(num_classes, pretrained, use_coord_att)
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


def create_fcos_model(num_classes=2, pretrained=True, device="cuda", use_coord_att=False):
    """Create and return FCOS model on specified device."""
    model = FCOSWrapper(num_classes=num_classes, pretrained=pretrained, use_coord_att=use_coord_att)
    model = model.to(device)
    return model
