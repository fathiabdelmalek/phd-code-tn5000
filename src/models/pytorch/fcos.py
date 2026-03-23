"""
FCOS model with optional Coordinate Attention injection.
"""

import torchvision
import torch

from . import register_model
from ..common.cord_att import CoordAtt


def _inject_coord_att(model):
    """Inject Coordinate Attention into ResNet backbone."""

    def make_coord_att_resnet():
        class CoordAttResNet(torch.nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
                self.coord_att = CoordAtt(512, 512)

            def forward(self, x):
                features = self.backbone(x)
                # Apply CA to last feature map
                features[-1] = self.coord_att(features[-1])
                return features

        return CoordAttResNet(model.backbone)

    # Replace backbone with CA-injected version
    model.backbone = torch.nn.Sequential(*list(model.backbone.children())[:-2])
    return model


@register_model("fcos", "fcos_ca")
class FCOSModel:
    """FCOS detector with optional Coordinate Attention."""

    def __init__(self, num_classes=2, use_coord_att=False, pretrained=True, **kwargs):
        self.num_classes = num_classes
        self.use_coord_att = use_coord_att
        self.pretrained = pretrained
        self.kwargs = kwargs

        # Build model
        self.model = torchvision.models.detection.fcos_resnet50_fpn(
            weights="DEFAULT" if pretrained else None, **kwargs
        )

        # Inject CoordAtt if requested
        if use_coord_att:
            self.model = _inject_coord_att(self.model)

        # Replace classification head for custom num_classes
        ch = self.model.head.classification_head
        in_channels = ch.conv[0].in_channels
        num_anchors = ch.num_anchors
        self.model.head.classification_head = (
            torchvision.models.detection.fcos.FCOSClassificationHead(
                in_channels, num_anchors, num_classes + 1
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
