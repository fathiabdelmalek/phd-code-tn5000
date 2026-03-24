"""
Simple Configuration - All hyperparameters in one place.
Edit this file to change training settings.
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.resolve()

PRESETS = {
    "fast": {
        "optimizer": "SGD",
        "lr": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 1,
    },
    "standard": {
        "optimizer": "SGD",
        "lr": 0.001,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
    },
    "adamw": {
        "optimizer": "AdamW",
        "lr": 0.0001,
        "weight_decay": 0.01,
        "warmup_epochs": 3,
    },
    "fine_tune": {
        "optimizer": "SGD",
        "lr": 0.0001,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "warmup_epochs": 5,
    },
}


class Config:
    def __init__(self, **kwargs):
        data_path = kwargs.get("data", "data/yolo/data.yaml")
        if not Path(data_path).is_absolute():
            self.data = str(BASE_DIR / data_path)
        else:
            self.data = data_path
        self.batch_size = kwargs.get("batch_size", 2)
        self.epochs = kwargs.get("epochs", 50)
        self.preset = kwargs.get("preset", "standard")
        self.device = kwargs.get("device", "cuda")
        self.img_size = kwargs.get("img_size", 640)

        # Load preset
        preset = PRESETS.get(self.preset, PRESETS.get("standard"))
        self.optimizer = kwargs.get("optimizer", preset.get("optimizer"))
        self.lr = kwargs.get("lr", preset.get("lr"))
        self.momentum = kwargs.get("momentum", preset.get("momentum", 0.937))
        self.weight_decay = kwargs.get("weight_decay", preset.get("weight_decay"))
        self.warmup_epochs = kwargs.get("warmup_epochs", preset.get("warmup_epochs"))
        self.augmentations = kwargs.get("augmentations", preset.get("augmentations"))

    def __repr__(self):
        return (
            f"Config(optimizer={self.optimizer}, lr={self.lr}, batch={self.batch_size})"
        )
