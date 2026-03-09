import os

from ultralytics import YOLO
import torchvision

from src.models.common.cord_att import CoordAtt


def get_model(model_name, weights=None, num_classes=2):
    """
    Unified Model Factory.
    - yolo26_ca: Custom YOLO with Coordinate Attention
    - faster_rcnn: Standard Faster R-CNN with ResNet50 backbone
    - fcos: Fully Convolutional One-Stage object detector
    """
    model_name = model_name.lower()

    if 'yolo' in model_name:
        import ultralytics.nn.tasks
        ultralytics.nn.tasks.CoordAtt = CoordAtt


        if weights and os.path.exists(weights):
            return YOLO(f'experiments/{model_name}/weights/best.pt')

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        yaml_path = os.path.join(project_root, 'models', 'cfg', f'{model_name}.yaml')
        print(f"DEBUG: Loading model config from: {yaml_path}")  # Good to keep for now
        return YOLO(yaml_path)

    elif 'rcnn' in model_name:
        # Standard Faster R-CNN from torchvision
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights if weights else "DEFAULT")
        # Replace the head for 2 classes (benign, malignant + background)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes + 1)
        return model

    elif 'fcos' in model_name:
        # FCOS: A great anchor-free alternative for medical imaging
        model = torchvision.models.detection.fcos_resnet50_fpn(weights=weights if weights else "DEFAULT", num_classes=num_classes)
        return model

    else:
        raise ValueError(f"Model {model_name} not supported in registry.")
