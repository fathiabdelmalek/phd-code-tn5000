import csv
import shutil
import torch
from tqdm import tqdm
from pathlib import Path

from .helpers import MetricsCalculator


class PyTorchTrainer:
    """Simple PyTorch trainer for FCOS, Faster-RCNN, etc."""

    COCO_TO_TN5000 = {
        1: 0,  # COCO person -> TN5000 benign
        3: 1,  # COCO bird -> TN5000 malignant
    }
    TN5000_TO_COCO = {v: k for k, v in COCO_TO_TN5000.items()}

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device
        self.model.to(self.device)
        self.exp_dir = getattr(config, "exp_dir", Path("experiments/default"))
        self.train_dir = self.exp_dir / "train"
        self.metrics_calculator = MetricsCalculator(iou_threshold=0.5)

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

    def _remap_predictions(self, predictions):
        """Remap COCO labels back to TN5000 labels in predictions."""
        remapped = []
        for pred in predictions:
            if isinstance(pred, dict) and "labels" in pred:
                new_pred = pred.copy()
                labels = new_pred["labels"]
                boxes = new_pred["boxes"]
                scores = new_pred["scores"]

                if isinstance(labels, torch.Tensor):
                    labels = labels.clone()

                    # First, filter to only COCO classes that map to TN5000
                    valid_mask = torch.zeros_like(labels, dtype=torch.bool)
                    for coco_label in self.COCO_TO_TN5000.keys():
                        valid_mask = valid_mask | (labels == coco_label)

                    # Keep only valid predictions
                    labels = labels[valid_mask]
                    boxes = boxes[valid_mask]
                    scores = scores[valid_mask]

                    # Remap remaining labels
                    for coco_label, tn_lbl in self.COCO_TO_TN5000.items():
                        labels[labels == coco_label] = tn_lbl

                    new_pred["labels"] = labels
                    new_pred["boxes"] = boxes
                    new_pred["scores"] = scores

                remapped.append(new_pred)
            else:
                remapped.append(pred)
        return remapped

    def _setup_logging(self):
        """Setup CSV logging and directories."""
        self.train_dir.mkdir(parents=True, exist_ok=True)
        (self.train_dir / "weights").mkdir(exist_ok=True)

        self.csv_path = self.train_dir / "results.csv"
        self.csv_columns = [
            "epoch",
            "time",
            "train/box_loss",
            "train/cls_loss",
            "train/dfl_loss",
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "val/box_loss",
            "val/cls_loss",
            "val/dfl_loss",
        ]

        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_columns)
                writer.writeheader()

    def _log_epoch(
        self,
        epoch,
        elapsed_time,
        train_loss,
        val_loss,
        val_cls_loss,
        val_box_loss,
        val_ctr_loss,
        metrics,
    ):
        """Log epoch metrics to CSV."""
        row = {
            "epoch": epoch + 1,
            "time": f"{elapsed_time:.4f}",
            "train/box_loss": f"{train_loss:.5f}",
            "train/cls_loss": f"{val_cls_loss:.5f}",
            "train/dfl_loss": f"{val_ctr_loss:.5f}",
            "metrics/precision(B)": f"{metrics.get('precision', 0):.5f}",
            "metrics/recall(B)": f"{metrics.get('recall', 0):.5f}",
            "metrics/mAP50(B)": f"{metrics.get('mAP50', 0):.5f}",
            "metrics/mAP50-95(B)": f"{metrics.get('mAP', 0):.5f}",
            "val/box_loss": f"{val_box_loss:.5f}",
            "val/cls_loss": f"{val_cls_loss:.5f}",
            "val/dfl_loss": f"{val_ctr_loss:.5f}",
        }
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_columns)
            writer.writerow(row)

        print(
            f"Epoch {epoch + 1}/{self.config.epochs}: box_loss={train_loss:.5f}, "
            f"cls_loss={val_cls_loss:.5f}, dfl_loss={val_ctr_loss:.5f}, mP={metrics.get('precision', 0):.4f}, "
            f"mR={metrics.get('recall', 0):.4f}, mAP50={metrics.get('mAP50', 0):.4f}, "
            f"mAP50-95={metrics.get('mAP', 0):.4f}"
        )

    def _save_weights(self, epoch, metrics, is_best):
        """Save model weights."""
        weights_dir = self.train_dir / "weights"
        torch.save(self.model.state_dict(), weights_dir / "last.pt")
        if is_best:
            torch.save(self.model.state_dict(), weights_dir / "best.pt")

    def _generate_plots(self):
        """Generate training plots from CSV."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            epochs = []
            train_losses = []
            val_losses = []
            maps50 = []
            maps = []
            precisions = []
            recalls = []

            with open(self.csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    epochs.append(int(row["epoch"]))
                    train_losses.append(float(row["train/box_loss"]))
                    val_losses.append(float(row["val/box_loss"]))
                    maps50.append(float(row["metrics/mAP50(B)"]))
                    maps.append(float(row["metrics/mAP50-95(B)"]))
                    precisions.append(float(row["metrics/precision(B)"]))
                    recalls.append(float(row["metrics/recall(B)"]))

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))

            axes[0, 0].plot(epochs, train_losses, label="Train Loss", marker="o")
            axes[0, 0].plot(epochs, val_losses, label="Val Loss", marker="o")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].set_title("Box Loss")
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            axes[0, 1].plot(epochs, maps50, label="mAP@50", marker="o")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("mAP@50")
            axes[0, 1].set_title("mAP@50")
            axes[0, 1].legend()
            axes[0, 1].grid(True)

            axes[0, 2].plot(epochs, maps, label="mAP@50-95", marker="o")
            axes[0, 2].set_xlabel("Epoch")
            axes[0, 2].set_ylabel("mAP@50-95")
            axes[0, 2].set_title("mAP@50-95")
            axes[0, 2].legend()
            axes[0, 2].grid(True)

            axes[1, 0].plot(epochs, precisions, label="Precision", marker="o")
            axes[1, 0].plot(epochs, recalls, label="Recall", marker="s")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Score")
            axes[1, 0].set_title("Precision/Recall")
            axes[1, 0].legend()
            axes[1, 0].grid(True)

            axes[1, 1].plot(epochs, precisions, label="Precision", marker="o")
            axes[1, 1].plot(epochs, recalls, label="Recall", marker="s")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Score")
            axes[1, 1].set_title("Precision-Recall Curve")
            axes[1, 1].legend()
            axes[1, 1].grid(True)

            f1s = [2 * p * r / (p + r + 1e-8) for p, r in zip(precisions, recalls)]
            axes[1, 2].plot(epochs, f1s, label="F1", marker="o", color="purple")
            axes[1, 2].set_xlabel("Epoch")
            axes[1, 2].set_ylabel("F1 Score")
            axes[1, 2].set_title("F1 Score")
            axes[1, 2].legend()
            axes[1, 2].grid(True)

            plt.tight_layout()
            plt.savefig(self.train_dir / "results.png", dpi=150)
            plt.close()

            print(f"Plots saved to {self.train_dir / 'results.png'}")
        except ImportError:
            print("matplotlib not available, skipping plots")

    def train(self):
        """Train PyTorch model with YOLO format dataloader and comprehensive metrics."""
        import time as time_module
        from src.data.voc_loader import create_tn5000_dataloaders

        base_dir = Path(__file__).parent.parent.resolve()
        data_root = base_dir / "data"

        self._setup_logging()

        # Use YOLO format (same as YOLO training)
        loaders = create_tn5000_dataloaders(
            data_root=str(data_root),
            batch_size=self.config.batch_size,
            num_workers=4,
            use_yolo_format=True,
        )
        train_loader = loaders["train"]  # Use train set back
        val_loader = loaders["val"]

        optimizer = self._build_optimizer()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        print(f"Training PyTorch model: {len(train_loader)} batches")
        print(f"FCOS pretrained on COCO, 2-class output for TN5000")
        print(f"Saving to: {self.train_dir}")

        best_map = -1
        best_epoch = 0

        for epoch in range(self.config.epochs):
            epoch_start = time_module.time()
            self.model.train()
            total_loss = 0
            train_losses = {"box_loss": 0, "cls_loss": 0, "ctr_loss": 0}

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            for images, targets in pbar:
                images = [img.to(self.device) for img in images]
                targets = [
                    {k: v.to(self.device) for k, v in t.items()} for t in targets
                ]
                # FCOS now outputs 2 classes directly, no need to remap labels

                optimizer.zero_grad()
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                # Check for NaN/Inf in loss
                if not torch.isfinite(losses):
                    print(
                        f"Warning: Non-finite loss at batch {pbar.n}: {losses.item()}"
                    )
                    print(f"  classification: {loss_dict['classification'].item()}")
                    print(f"  bbox_regression: {loss_dict['bbox_regression'].item()}")
                    print(f"  bbox_ctrness: {loss_dict['bbox_ctrness'].item()}")
                    optimizer.zero_grad()
                    continue

                losses.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += losses.item()
                train_losses["box_loss"] += loss_dict.get("bbox_regression", 0).item()
                train_losses["cls_loss"] += loss_dict.get("classification", 0).item()
                train_losses["ctr_loss"] += loss_dict.get("bbox_ctrness", 0).item()
                pbar.set_postfix({"loss": total_loss / (pbar.n + 1)})

            # Skip epoch if all losses were NaN
            if not torch.isfinite(torch.tensor(train_losses["cls_loss"])):
                print(f"Warning: All training losses were NaN, skipping epoch {epoch}")
                continue

            train_loss = total_loss / len(train_loader)
            train_cls_loss = train_losses["cls_loss"] / len(train_loader)
            train_box_loss = train_losses["box_loss"] / len(train_loader)
            train_ctr_loss = train_losses["ctr_loss"] / len(train_loader)
            scheduler.step()

            # Validation with metrics
            self.model.train()
            val_loss = 0
            train_losses = {"box_loss": 0, "cls_loss": 0, "ctr_loss": 0}
            self.metrics_calculator.reset()

            with torch.no_grad():
                for images, targets in tqdm(val_loader, desc="Evaluating"):
                    images = [img.to(self.device) for img in images]
                    targets = [
                        {k: v.to(self.device) for k, v in t.items()} for t in targets
                    ]
                    # FCOS now outputs 2 classes directly, no remapping needed
                    loss_dict = self.model(images, targets)

                    if isinstance(loss_dict, dict):
                        val_loss += sum(
                            v.item() if hasattr(v, "item") else v
                            for v in loss_dict.values()
                        )
                        # Accumulate individual losses
                        train_losses["box_loss"] += loss_dict.get(
                            "bbox_regression", 0
                        ).item()
                        train_losses["cls_loss"] += loss_dict.get(
                            "classification", 0
                        ).item()
                        train_losses["ctr_loss"] += loss_dict.get(
                            "bbox_ctrness", 0
                        ).item()
                    elif isinstance(loss_dict, list):
                        val_loss += sum(
                            v.item() if hasattr(v, "item") else v for v in loss_dict
                        )

                    # Get predictions for metrics (no remapping needed)
                    self.model.eval()
                    predictions = self.model(images)
                    self.metrics_calculator.evaluate_batch(predictions, targets)
                    self.model.train()

            val_loss = val_loss / len(val_loader)
            val_cls_loss = train_losses["cls_loss"] / len(val_loader)
            val_box_loss = train_losses["box_loss"] / len(val_loader)
            val_ctr_loss = train_losses["ctr_loss"] / len(val_loader)
            metrics = self.metrics_calculator.compute_metrics()

            elapsed_time = time_module.time() - epoch_start

            # Log and save
            self._log_epoch(
                epoch,
                elapsed_time,
                train_loss,
                val_loss,
                val_cls_loss,
                val_box_loss,
                val_ctr_loss,
                metrics,
            )

            is_best = metrics.get("mAP", 0) > best_map
            if is_best:
                best_map = metrics.get("mAP", 0)
                best_epoch = epoch

            self._save_weights(epoch, metrics, is_best)

        # Generate plots
        self._generate_plots()

        print(f"\nTraining complete!")
        print(f"Best mAP: {best_map:.4f} at epoch {best_epoch}")
        print(f"Results saved to: {self.train_dir}")

        # Copy to val weights
        val_weights = self.exp_dir / "val" / "weights"
        val_weights.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.train_dir / "weights" / "best.pt", val_weights / "best.pt")
        shutil.copy2(self.train_dir / "weights" / "last.pt", val_weights / "last.pt")

        return {
            "best_map": best_map,
            "best_epoch": best_epoch,
            "train_dir": str(self.train_dir),
        }
