import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path
from copy import deepcopy


class Config:
    """Configuration manager with dot-notation access.

    Supports loading from YAML files and merging with overrides.
    """

    def __init__(self, data: Optional[Dict] = None):
        self._data = data or {}

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            return cls(yaml.safe_load(f))

    @classmethod
    def from_defaults(cls) -> "Config":
        """Load default configuration."""
        defaults_path = (
            Path(__file__).parent.parent.parent / "configs" / "defaults.yaml"
        )
        if defaults_path.exists():
            return cls.from_yaml(str(defaults_path))
        return cls(cls._get_default_config())

    @classmethod
    def _get_default_config(cls) -> Dict:
        """Return the default configuration dictionary."""
        return {
            "training": {
                "epochs": 50,
                "batch_size": 8,
                "optimizer": "AdamW",
                "lr0": 0.001,
                "lrf": 0.01,
                "weight_decay": 0.001,
                "momentum": 0.937,
                "warmup_epochs": 3,
                "warmup_momentum": 0.8,
                "warmup_bias_lr": 0.1,
                "cos_lr": True,
                "patience": 20,
                "gradient_clip": 10.0,
                "mixed_precision": False,
            },
            "loss": {
                "box": 10.0,
                "cls": 1.5,
                "dfl": 2.0,
            },
            "augmentation": {
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
                "degrees": 0.0,
                "translate": 0.1,
                "scale": 0.5,
                "shear": 0.0,
                "perspective": 0.0,
                "flipud": 0.0,
                "fliplr": 0.5,
                "mosaic": 1.0,
                "mixup": 0.0,
                "copy_paste": 0.0,
            },
            "evaluation": {
                "conf_thres": 0.25,
                "iou_thres": 0.45,
                "max_det": 300,
                "plots": True,
            },
            "model": {
                "imgsz": 640,
                "conf": 0.25,
                "device": "",
            },
        }

    def merge(self, overrides: Dict) -> "Config":
        """Merge overrides into the current config."""
        merged = deepcopy(self._data)
        self._deep_update(merged, overrides)
        return Config(merged)

    def _deep_update(self, base: Dict, updates: Dict) -> None:
        """Recursively update nested dictionary."""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value using dot notation (e.g., 'training.lr0')."""
        keys = key.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    def __getitem__(self, key: str) -> Any:
        """Get a value using dot notation."""
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        """Check if a key exists."""
        return self.get(key) is not None

    def to_dict(self) -> Dict:
        """Return the config as a dictionary."""
        return deepcopy(self._data)

    def save(self, path: str) -> None:
        """Save config to a YAML file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self._data, f, default_flow_style=False)

    def __repr__(self) -> str:
        return f"Config({self._data})"


def load_config(
    config_path: Optional[str] = None, overrides: Optional[Dict] = None
) -> Config:
    """Load configuration from file or defaults, with optional overrides.

    Args:
        config_path: Path to YAML config file. If None, uses defaults.
        overrides: Dictionary of values to override.

    Returns:
        Config instance with merged settings.
    """
    if config_path and os.path.exists(config_path):
        config = Config.from_yaml(config_path)
    else:
        config = Config.from_defaults()

    if overrides:
        config = config.merge(overrides)

    return config
