"""
Trainers - Simple, self-contained training classes.
"""

import json
import shutil
import torch
from tqdm import tqdm
from pathlib import Path


class YOLOTrainer:
    """Simple YOLO trainer."""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device

    def train(self):
        """Train YOLO model."""
        exp_dir = getattr(self.config, "exp_dir", Path("experiments/default"))
        train_dir = exp_dir / "train"

        # Ensure absolute path - YOLO requires this to avoid adding 'runs/' prefix
        exp_dir = exp_dir.resolve()
        train_dir = train_dir.resolve()

        print(f"Training YOLO: {self.config.epochs} epochs")
        print(f"Saving to: {train_dir}")

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
            augment=True,
            exist_ok=True,
            plots=True,
            save=True,
            project=str(exp_dir),
            name="train",
        )

        # Copy weights to val/weights/ if they exist in train/
        val_weights = exp_dir / "val" / "weights"
        val_weights.mkdir(parents=True, exist_ok=True)

        for weight_file in ["best.pt", "last.pt"]:
            src = train_dir / "weights" / weight_file
            if src.exists():
                shutil.copy2(src, val_weights / weight_file)

        return results


class PyTorchTrainer:
    """Simple PyTorch trainer for FCOS, Faster-RCNN, etc."""

    COCO_TO_TN5000 = {
        1: 0,  # COCO person -> TN5000 benign
        3: 1,  # COCO car -> TN5000 malignant (or use bird 3)
    }
    TN5000_TO_COCO = {v: k for k, v in COCO_TO_TN5000.items()}

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device
        self.model.to(self.device)

    def _build_optimizer(self):
        """Build optimizer."""
        if self.config.optimizer == "AdamW":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "SGD":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.lr,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "Adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _remap_labels(self, targets):
        """Remap TN5000 labels to COCO labels for pretrained FCOS."""
        remapped = []
        for t in targets:
            new_t = t.copy()
            labels = t["labels"].clone()
            for tn5000_label, coco_label in self.TN5000_TO_COCO.items():
                labels[labels == tn5000_label] = coco_label
            new_t["labels"] = labels
            remapped.append(new_t)
        return remapped

    def train(self):
        """Train PyTorch model with VOC dataloader."""
        from data.voc_loader import create_tn5000_dataloaders

        base_dir = Path(__file__).parent.parent.resolve()
        data_root = base_dir / "data" / "voc"

        loaders = create_tn5000_dataloaders(
            data_root=str(data_root),
            batch_size=self.config.batch_size,
            num_workers=4,
        )
        train_loader = loaders["train"]
        val_loader = loaders["val"]

        optimizer = self._build_optimizer()

        print(f"Training PyTorch model: {len(train_loader)} batches")
        print(f"FCOS pretrained on COCO, remapping TN5000 labels to COCO labels")

        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
            for images, targets in pbar:
                images = [img.to(self.device) for img in images]
                targets = [
                    {k: v.to(self.device) for k, v in t.items()} for t in targets
                ]
                targets = self._remap_labels(targets)

                optimizer.zero_grad()
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()

                total_loss += losses.item()
                pbar.set_postfix({"loss": total_loss / (pbar.n + 1)})

            # Validation - FCOS returns detections in eval mode, so we use train mode
            self.model.train()
            val_loss = 0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = [img.to(self.device) for img in images]
                    targets = [
                        {k: v.to(self.device) for k, v in t.items()} for t in targets
                    ]
                    targets = self._remap_labels(targets)
                    loss_dict = self.model(images, targets)
                    if isinstance(loss_dict, dict):
                        val_loss += sum(
                            v.item() if hasattr(v, "item") else v
                            for v in loss_dict.values()
                        )
                    elif isinstance(loss_dict, list):
                        val_loss += sum(
                            v.item() if hasattr(v, "item") else v for v in loss_dict
                        )

            print(
                f"Epoch {epoch}: train_loss={total_loss / len(train_loader):.4f}, val_loss={val_loss / len(val_loader):.4f}"
            )

        return {"train_loss": total_loss / len(train_loader)}
