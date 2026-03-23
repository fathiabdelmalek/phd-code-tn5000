"""
PyTorch Model Registry - Auto-discovery plugin system.

Usage:
    from src.models.pytorch import register_model

    @register_model('my_model')
    class MyCustomModel:
        def __init__(self, num_classes=2, **kwargs):
            ...
"""

import os
from glob import glob

MODEL_REGISTRY = {}


def register_model(*names):
    """
    Decorator to register PyTorch models.

    Usage:
        @register_model('fcos')
        @register_model('fcos_ca')  # Alias
        class FCOSModel:
            ...
    """

    def decorator(cls):
        for name in names:
            MODEL_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def list_models():
    """List all registered model names."""
    return sorted(MODEL_REGISTRY.keys())


def get_model_class(name):
    """Get a model class by name."""
    name_lower = name.lower()
    if name_lower in MODEL_REGISTRY:
        return MODEL_REGISTRY[name_lower]
    raise ValueError(f"Unknown model: {name}. Available: {list_models()}")


# Auto-import all model files
_models_dir = os.path.dirname(__file__)
for _f in glob(os.path.join(_models_dir, "*.py")):
    if not _f.endswith("__init__.py"):
        _module_name = os.path.basename(_f)[:-3]
        try:
            __import__(
                f"src.models.pytorch.{_module_name}", fromlist=["register_model"]
            )
        except ImportError as e:
            print(f"Warning: Could not import {_module_name}: {e}")

__all__ = ["register_model", "list_models", "get_model_class", "MODEL_REGISTRY"]
