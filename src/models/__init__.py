"""
Models - Simple model loading.

Usage:
    from models import get_model

    # YOLO with scale parameter
    model = get_model('yolo26', scale='n')  # nano
    model = get_model('yolo26', scale='s')  # small
    model = get_model('yolo26', scale='m')  # medium
    model = get_model('yolo26', scale='l')  # large
    model = get_model('yolo26', scale='x')  # extra-large

    # FCOS
    model = get_model('fcos')
"""

from .yolo import get_yolo_model
from .fcos import FCOSWrapper, get_fcos_model


# Available scales for YOLO26
YOLO_SCALES = ["n", "s", "m", "l", "x"]

# Model scale descriptions
YOLO_SCALE_INFO = {
    "n": "nano (~2.5M params)",
    "s": "small (~10M params)",
    "m": "medium (~22M params)",
    "l": "large (~26M params)",
    "x": "extra-large (~59M params)",
}


def get_model(
    name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    scale: str = "n",
    use_coord_att: bool = False,
):
    """
    Get model by name.

    Args:
        name: Model name ('yolo26', 'fcos', 'fcnn')
        num_classes: Number of classes
        pretrained: Use pretrained weights (YOLO only)
        scale: YOLO scale - 'n', 's', 'm', 'l', or 'x' (default: 'n')
        use_coord_att: Use CoordAtt in FCOS (default: False)

    Returns:
        Model instance
    """
    name = name.lower()

    if name.startswith("yolo26"):
        return get_yolo_model(name="yolo26", num_classes=num_classes, scale=scale)
    elif name == "fcos":
        return FCOSWrapper(num_classes, pretrained, use_coord_att)
    else:
        raise ValueError(f"Unknown model: {name}. Available: ['yolo26', 'fcos']")


__all__ = [
    "get_model",
    "get_yolo_model",
    "get_fcos_model",
    "FCOSWrapper",
    "YOLO_SCALES",
    "YOLO_SCALE_INFO",
]
