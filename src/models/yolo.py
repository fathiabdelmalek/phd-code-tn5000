"""
YOLO Model - Simple wrapper for Ultralytics YOLO.
"""

from pathlib import Path
from ultralytics import YOLO
import ultralytics.nn.tasks


def get_yolo_model(name: str = "yolo26", num_classes: int = 2, scale: str = "n"):
    """
    Load YOLO model.

    Args:
        name: Model name (default: yolo26)
        num_classes: Number of classes
        scale: Model scale - 'n', 's', 'm', 'l', or 'x' (default: 'n')
    """
    # Register CoordAtt
    from .common import CoordAtt

    ultralytics.nn.tasks.CoordAtt = CoordAtt

    # Find config file
    cfg_dir = Path(__file__).parent / "cfg"
    # Try both with _ca and without
    cfg_path = cfg_dir / f"{name}.yaml"
    if not cfg_path.exists():
        cfg_path = cfg_dir / f"{name}_ca.yaml"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Model config not found: {cfg_path}")

    # Load and modify config
    import yaml

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    cfg["nc"] = num_classes

    # Validate scale
    available_scales = list(cfg.get("scales", {}).keys())
    if scale not in available_scales:
        raise ValueError(f"Invalid scale '{scale}'. Available: {available_scales}")

    # Keep only the selected scale in the yaml (required for ultralytics to pick it up)
    cfg["scales"] = {scale: cfg["scales"][scale]}

    # Save temp config with scale in filename
    import tempfile

    temp_path = f"temp_yolo26{scale}.yaml"
    with open(temp_path, "w") as f:
        yaml.dump(cfg, f)

    # Load model
    model = YOLO(temp_path)

    # Cleanup temp file
    import os

    os.remove(temp_path)

    return model
