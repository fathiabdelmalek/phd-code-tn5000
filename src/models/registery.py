"""
Unified Model Factory.

Auto-discovers:
- YOLO models from src/models/cfg/*.yaml
- PyTorch models from src/models/pytorch/ (using @register_model decorator)

Usage:
    from src.models import get_model

    # YOLO
    model = get_model('yolo26n_ca')

    # PyTorch
    model = get_model('fcos', num_classes=2)
    model = get_model('fcos_ca', use_coord_att=True)  # With CA injection
"""

import os
import torch
from glob import glob

# YOLO imports
from ultralytics import YOLO

# Common modules
from .common.cord_att import CoordAtt as CoordAttOriginal


class CoordAtt(CoordAttOriginal):
    """Wrapper for CoordAtt that matches Ultralytics expected signature."""

    def __init__(self, inp, oup=None, reduction=32):
        if oup is None:
            oup = inp
        super().__init__(inp, oup, reduction)


def _discover_yolo_configs():
    """Auto-discover YOLO configs from cfg/ directory."""
    cfg_dir = os.path.join(os.path.dirname(__file__), "cfg")
    configs = {}
    for yaml_file in glob(os.path.join(cfg_dir, "*.yaml")):
        model_name = os.path.basename(yaml_file).replace(".yaml", "")
        configs[model_name.lower()] = yaml_file
    return configs


def _discover_pytorch_models():
    """Import and discover PyTorch models."""
    from .pytorch import MODEL_REGISTRY

    return MODEL_REGISTRY


YOLO_CONFIGS = _discover_yolo_configs()


def get_model(model_name, weights=None, num_classes=2, **kwargs):
    """
    Unified Model Factory.

    Args:
        model_name: Model name (e.g., 'yolo26n_ca', 'fcos', 'fcos_ca', 'faster_rcnn')
        weights: Path to weights file or None
        num_classes: Number of classes (default: 2 for thyroid: benign/malignant)
        **kwargs: Additional model-specific arguments

    Returns:
        Model instance
    """
    model_name_lower = model_name.lower()

    # Try YOLO first
    if model_name_lower in YOLO_CONFIGS:
        return _create_yolo_model(
            model_name_lower, weights, YOLO_CONFIGS[model_name_lower]
        )

    # Try PyTorch models
    pytorch_models = _discover_pytorch_models()
    if model_name_lower in pytorch_models:
        return _create_pytorch_model(
            model_name_lower,
            pytorch_models[model_name_lower],
            num_classes,
            weights,
            **kwargs,
        )

    # Available models
    available = list(YOLO_CONFIGS.keys()) + list(pytorch_models.keys())
    raise ValueError(
        f"Unknown model: {model_name}. Available: {sorted(set(available))}"
    )


def _create_yolo_model(model_name, weights, yaml_path):
    """Create a YOLO model."""
    import ultralytics.nn.tasks

    ultralytics.nn.tasks.CoordAtt = CoordAtt

    if weights and os.path.exists(weights):
        return YOLO(weights)

    print(f"Loading YOLO model from: {yaml_path}")
    return YOLO(yaml_path)


def _create_pytorch_model(model_name, model_class, num_classes, weights, **kwargs):
    """Create a PyTorch model."""
    model = model_class(num_classes=num_classes, **kwargs)

    if weights and os.path.exists(weights):
        model.load_state_dict(torch.load(weights, map_location="cpu"))

    return model


def list_available_models():
    """List all available model names."""
    yolo_models = list(YOLO_CONFIGS.keys())
    pytorch_models = list(_discover_pytorch_models().keys())
    return {
        "yolo": yolo_models,
        "pytorch": pytorch_models,
        "all": sorted(set(yolo_models + pytorch_models)),
    }
