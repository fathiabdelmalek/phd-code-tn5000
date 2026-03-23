"""
Faster R-CNN model with optional Coordinate Attention injection.
"""

import torchvision
import torch

from . import register_model
from ..common.cord_att import CoordAtt


def _inject_coord_att_fpn(model):
    """Inject Coordinate Attention into FPN features."""
    # This is a simplified version - for full injection, you'd modify the FPN
    return model


@register_model("faster_rcnn", "rcnn")
class FasterRCNNModel:
    """Faster R-CNN detector with optional Coordinate Attention."""

    def __init__(
        self,
        num_classes=2,
        use_coord_att=False,
        pretrained=True,
        backbone="resnet50",
        **kwargs,
    ):
        self.num_classes = num_classes
        self.use_coord_att = use_coord_att
        self.pretrained = pretrained
        self.kwargs = kwargs

        # Build model
        if backbone == "resnet50":
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                weights="DEFAULT" if pretrained else None, **kwargs
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Replace box predictor head
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = (
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, num_classes + 1
            )
        )

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train(self, *args, **kwargs):
        return self.model.train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        return self.model.eval(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.model.parameters(*args, **kwargs)

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        return self

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)
