from pathlib import Path

from src.transforms import get_train_transforms


class YOLOTrainer:
    """Simple YOLO trainer."""

    def __init__(self, model, config):
        self.model = model
        self.config = config

    def train(self):
        """Train YOLO models."""
        exp_dir = getattr(self.config, "exp_dir", Path("experiments/default"))
        train_dir = exp_dir / "train"

        # Ensure absolute path - YOLO requires this to avoid adding 'runs/' prefix
        exp_dir = exp_dir.resolve()
        train_dir = train_dir.resolve()

        print(f"Training YOLO: {self.config.epochs} epochs")
        print(f"Saving to: {train_dir}")

        # Augmentations
        augmentations = (
            get_train_transforms('yolo', self.config.augmentations)
            if self.config.augmentations != 'none'
            else None
        )

        # YOLO training
        # IMPORTANT: Use absolute paths for project to avoid 'runs/' prefix
        results = self.model.train(
            data=self.config.data,
            epochs=self.config.epochs,
            batch=self.config.batch_size,
            imgsz=self.config.img_size,
            device=self.config.device,
            optimizer=self.config.optimizer,
            lr0=self.config.lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
            warmup_epochs=self.config.warmup_epochs,
            augmentations=augmentations,
            patience=self.config.patience,
            exist_ok=True,
            plots=True,
            save=True,
            project=str(exp_dir),
            name="train",
        )

        return results
