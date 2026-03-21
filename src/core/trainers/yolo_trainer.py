import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch

from ..base_trainer import BaseTrainer
from ...models.registery import get_model
from ...datasets.factory import get_dataloader


class YOLOTrainer(BaseTrainer):
    """Trainer implementation for YOLO models using Ultralytics."""

    def __init__(
        self,
        model_name: str,
        data_root: str,
        exp_dir: str,
        config: Optional[Dict] = None,
        weights: Optional[str] = None,
        resume: bool = False,
        callbacks: Optional[Dict] = None,
    ):
        self.model_name = model_name
        self.data_root = data_root
        self.weights = weights
        self.resume = resume

        train_loader = get_dataloader(model_name, data_root, split="train")
        val_loader = get_dataloader(model_name, data_root, split="val")

        super().__init__(
            model=None,
            train_loader=train_loader,
            val_loader=val_loader,
            exp_dir=exp_dir,
            config=config or {},
            callbacks=callbacks,
        )

        self._build_model()

    def _build_model(self):
        """Build the YOLO model."""
        from ultralytics import YOLO
        from ...models.common.cord_att import CoordAtt

        import ultralytics.nn.tasks

        ultralytics.nn.tasks.CoordAtt = CoordAtt

        if self.resume:
            last_ckpt = str(self.exp_dir / "weights" / "last.pt")
            if os.path.exists(last_ckpt):
                print(f"Resuming from: {last_ckpt}")
                self.model = YOLO(last_ckpt)
            else:
                raise FileNotFoundError(f"Resume checkpoint not found: {last_ckpt}")
        elif self.weights and os.path.exists(self.weights):
            self.model = YOLO(self.weights)
        else:
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            yaml_path = os.path.join(
                project_root, "models", "cfg", f"{self.model_name}.yaml"
            )
            if not os.path.exists(yaml_path):
                raise FileNotFoundError(f"Model config not found: {yaml_path}")
            self.model = YOLO(yaml_path)

    def train(self) -> Dict[str, Any]:
        """Execute the full training loop."""
        self._run_callback("on_train_start")
        self._save_experiment_metadata()

        abs_exp_dir = os.path.abspath(self.exp_dir)

        train_kwargs = {
            "data": self.train_loader,
            "batch": self.config.get("batch_size", 8),
            "epochs": self.config.get("epochs", 50),
            "project": os.path.dirname(abs_exp_dir),
            "name": os.path.basename(abs_exp_dir),
            "exist_ok": True,
            "plots": True,
            "save": True,
        }

        if self.resume:
            train_kwargs["resume"] = True
            results = self.model.train(**train_kwargs)
        else:
            train_kwargs.update(
                {
                    "optimizer": self.config.get("optimizer", "AdamW"),
                    "lr0": self.config.get("lr0", 0.001),
                    "lrf": self.config.get("lrf", 0.01),
                    "weight_decay": self.config.get("weight_decay", 0.001),
                    "cos_lr": self.config.get("cos_lr", True),
                    "patience": self.config.get("patience", 20),
                    "box": self.config.get("box", 10.0),
                    "cls": self.config.get("cls", 1.5),
                    "dfl": self.config.get("dfl", 2.0),
                }
            )
            results = self.model.train(**train_kwargs)

        self._run_callback("on_train_end", results)
        return self._extract_metrics(results)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        raise NotImplementedError("YOLO handles epoch training internally")

    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.model is None:
            raise RuntimeError("Model not initialized. Call train() first.")

        results = self.model.val(
            data=self.val_loader,
            split="val",
            plots=False,
        )

        return {
            "map50": float(results.box.map50),
            "map50_95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
        }

    def save_checkpoint(self, filepath: str, include_optimizer: bool = True) -> None:
        """Save checkpoint (YOLO handles this internally)."""
        pass

    def load_checkpoint(self, filepath: str) -> None:
        """Load checkpoint."""
        if self.model is not None:
            self.model = self.model.load(filepath)

    def _extract_metrics(self, results) -> Dict[str, Any]:
        """Extract metrics from YOLO results."""
        metrics = {}
        if hasattr(results, "results_dict"):
            metrics = results.results_dict
        elif hasattr(results, "box"):
            metrics = {
                "map50": float(results.box.map50),
                "map50_95": float(results.box.map),
                "precision": float(results.box.mp),
                "recall": float(results.box.mr),
            }
        return metrics
