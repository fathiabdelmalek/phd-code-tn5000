from typing import Dict, Optional, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..base_trainer import BaseTrainer
from ..checkpoint_manager import CheckpointManager


class PyTorchTrainer(BaseTrainer):
    """Trainer implementation for PyTorch models (Faster-RCNN, FCOS, etc.)."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        exp_dir: str,
        config: Optional[Dict] = None,
        device: str = "cuda",
        callbacks: Optional[Dict] = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint_manager = CheckpointManager(exp_dir)

        super().__init__(
            model=model.to(self.device),
            train_loader=train_loader,
            val_loader=val_loader,
            exp_dir=exp_dir,
            config=config or {},
            callbacks=callbacks,
        )

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = None

        if self.config.get("mixed_precision", False):
            self.scaler = torch.cuda.amp.GradScaler()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build the optimizer based on config."""
        optimizer_name = self.config.get("optimizer", "AdamW")
        lr = self.config.get("lr0", 0.001)
        weight_decay = self.config.get("weight_decay", 0.001)

        if optimizer_name == "AdamW":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "SGD":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=self.config.get("momentum", 0.937),
                weight_decay=weight_decay,
            )
        elif optimizer_name == "Adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _build_scheduler(self):
        """Build the learning rate scheduler."""
        epochs = self.config.get("epochs", 50)

        if self.config.get("cos_lr", True):
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=self.config.get("lrf", 0.01) * self.config.get("lr0", 0.001),
            )
        else:
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1,
            )

    def train(self) -> Dict[str, Any]:
        """Execute the full training loop."""
        self._run_callback("on_train_start")
        self._save_experiment_metadata()

        epochs = self.config.get("epochs", 50)

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            self.scheduler.step()

            should_continue = self.on_epoch_end(epoch, train_metrics, val_metrics)

            if not should_continue:
                break

        self._run_callback("on_train_end", self.val_history)
        return val_metrics

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for images, targets in pbar:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                self.scaler.scale(losses).backward()

                if self.config.get("gradient_clip", 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config["gradient_clip"]
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                losses.backward()

                if self.config.get("gradient_clip", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config["gradient_clip"]
                    )

                self.optimizer.step()

            total_loss += losses.item()
            num_batches += 1

            pbar.set_postfix({"loss": total_loss / num_batches})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {"loss": avg_loss}

    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validation"):
                images = [img.to(self.device) for img in images]
                targets = [
                    {k: v.to(self.device) for k, v in t.items()} for t in targets
                ]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                total_loss += losses.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {"val_loss": avg_loss}

    def save_checkpoint(self, filepath: str, include_optimizer: bool = True) -> None:
        """Save a checkpoint."""
        self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            metrics={"best_metric": self.best_metric},
            filepath=filepath,
            include_optimizer=include_optimizer,
        )

    def load_checkpoint(self, filepath: str) -> None:
        """Load a checkpoint."""
        info = self.checkpoint_manager.load(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            filepath=filepath,
            device=str(self.device),
        )
        self.current_epoch = info.get("epoch", 0) + 1
        if "metrics" in info and "best_metric" in info["metrics"]:
            self.best_metric = info["metrics"]["best_metric"]
