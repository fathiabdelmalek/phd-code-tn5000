import os

from ultralytics import YOLO
import torchvision
import torch
import torch.nn as nn

from src.models.common.cord_att import CoordAtt as CoordAttOriginal


class CoordAtt(CoordAttOriginal):
    """Wrapper for CoordAtt that matches Ultralytics expected signature."""

    def __init__(self, inp, oup=None, reduction=32):
        if oup is None:
            oup = inp
        super().__init__(inp, oup, reduction)


def get_model(model_name, weights=None, num_classes=2):
    """
    Unified Model Factory.
    - yolo26_ca: Custom YOLO with Coordinate Attention
    - faster_rcnn: Standard Faster R-CNN with ResNet50 backbone
    - fcos: Fully Convolutional One-Stage object detector
    """
    model_name = model_name.lower()

    if "yolo" in model_name:
        import ultralytics.nn.tasks

        ultralytics.nn.tasks.CoordAtt = CoordAtt

        if weights and os.path.exists(weights):
            return YOLO(f"experiments/{model_name}/weights/best.pt")

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        yaml_path = os.path.join(project_root, "models", "cfg", f"{model_name}.yaml")
        print(f"DEBUG: Loading model config from: {yaml_path}")
        return YOLO(yaml_path)

    elif "rcnn" in model_name:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=weights if weights else "DEFAULT"
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = (
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, num_classes + 1
            )
        )
        return model

    elif "fcos" in model_name:
        model = torchvision.models.detection.fcos_resnet50_fpn(
            weights=weights if weights else "DEFAULT"
        )

        in_channels = model.head.classification_head.conv[0].in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = (
            torchvision.models.detection.fcos.FCOSClassificationHead(
                in_channels, num_anchors, num_classes + 1
            )
        )

        return model

    else:
        raise ValueError(f"Model {model_name} not supported in registry.")
