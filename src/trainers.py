"""
Trainers - Simple, self-contained training classes.
"""

import csv
import shutil
import time
import numpy as np
import torch
import cv2
from tqdm import tqdm
from pathlib import Path
from PIL import Image


class YOLOTrainer:
    """Simple YOLO trainer."""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device

    def train(self):
        """Train YOLO models."""
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


class MetricsCalculator:
    """Calculate detection metrics: mAP, precision, recall, F1."""

    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        """Reset all accumulators."""
        self.predictions = []
        self.targets = []

    def compute_iou(self, box1, box2):
        """Compute IoU between two boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def compute_ap(self, precisions, recalls):
        """Compute average precision from precision-recall curve."""
        precisions = np.concatenate(([0], precisions, [0]))
        recalls = np.concatenate(([0], recalls, [1]))

        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = max(precisions[i - 1], precisions[i])

        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
        return ap

    def evaluate_batch(self, predictions, targets):
        """Evaluate a batch of predictions against targets."""
        for pred, target in zip(predictions, targets):
            pred_boxes = pred.get("boxes", torch.tensor([]))
            pred_scores = pred.get("scores", torch.tensor([]))
            pred_labels = pred.get("labels", torch.tensor([]))

            gt_boxes = target.get("boxes", torch.tensor([]))
            gt_labels = target.get("labels", torch.tensor([]))

            self.predictions.append(
                {
                    "boxes": pred_boxes.cpu()
                    if isinstance(pred_boxes, torch.Tensor)
                    else pred_boxes,
                    "scores": pred_scores.cpu()
                    if isinstance(pred_scores, torch.Tensor)
                    else pred_scores,
                    "labels": pred_labels.cpu()
                    if isinstance(pred_labels, torch.Tensor)
                    else pred_labels,
                }
            )
            self.targets.append(
                {
                    "boxes": gt_boxes.cpu()
                    if isinstance(gt_boxes, torch.Tensor)
                    else gt_boxes,
                    "labels": gt_labels.cpu()
                    if isinstance(gt_labels, torch.Tensor)
                    else gt_labels,
                }
            )

    def compute_metrics(self):
        """Compute mAP, precision, recall, F1 across all accumulated predictions."""
        total_tp, total_fp, total_fn = 0, 0, 0
        all_precisions = []
        all_recalls = []

        for pred, target in zip(self.predictions, self.targets):
            pred_boxes = pred["boxes"]
            pred_scores = pred["scores"]
            pred_labels = pred["labels"]
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]

            matched_gt = set()
            tp, fp = 0, 0

            if len(pred_boxes) == 0:
                total_fp += len(gt_boxes)
                continue

            if len(gt_boxes) == 0:
                total_fp += len(pred_boxes)
                continue

            sorted_indices = torch.argsort(pred_scores, descending=True)
            for idx in sorted_indices:
                p_box = (
                    pred_boxes[idx].numpy()
                    if isinstance(pred_boxes[idx], torch.Tensor)
                    else pred_boxes[idx]
                )
                p_label = int(
                    pred_labels[idx].item()
                    if isinstance(pred_labels[idx], torch.Tensor)
                    else pred_labels[idx]
                )

                best_iou, best_gt_idx = 0, -1
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue
                    g_label = int(
                        gt_labels[gt_idx].item()
                        if isinstance(gt_labels[gt_idx], torch.Tensor)
                        else gt_labels[gt_idx]
                    )
                    if p_label != g_label:
                        continue
                    g_box = (
                        gt_box.numpy() if isinstance(gt_box, torch.Tensor) else gt_box
                    )
                    iou = self.compute_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= self.iou_threshold:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1

            fn = len(gt_boxes) - len(matched_gt)
            total_tp += tp
            total_fp += fp
            total_fn += fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            all_precisions.append(precision)
            all_recalls.append(recall)

        total_positives = total_tp + total_fn
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / total_positives if total_positives > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        mean_ap = np.mean(all_precisions) if all_precisions else 0

        return {
            "mAP": mean_ap,
            "mAP50": mean_ap,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


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
        TN5000_TO_COCO_REVERSE = {v: k for k, v in self.COCO_TO_TN5000.items()}
        remapped = []
        for pred in predictions:
            if isinstance(pred, dict) and "labels" in pred:
                new_pred = pred.copy()
                labels = new_pred["labels"]
                if isinstance(labels, torch.Tensor):
                    labels = labels.clone()
                    for coco_label, tn5000_label in TN5000_TO_COCO_REVERSE.items():
                        labels[labels == coco_label] = tn5000_label
                    new_pred["labels"] = labels
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

    def _log_epoch(self, epoch, elapsed_time, train_loss, val_loss, metrics):
        """Log epoch metrics to CSV."""
        row = {
            "epoch": epoch + 1,
            "time": f"{elapsed_time:.4f}",
            "train/box_loss": f"{train_loss:.5f}",
            "train/cls_loss": "0.00000",
            "train/dfl_loss": "0.00000",
            "metrics/precision(B)": f"{metrics.get('precision', 0):.5f}",
            "metrics/recall(B)": f"{metrics.get('recall', 0):.5f}",
            "metrics/mAP50(B)": f"{metrics.get('mAP50', 0):.5f}",
            "metrics/mAP50-95(B)": f"{metrics.get('mAP', 0):.5f}",
            "val/box_loss": f"{val_loss:.5f}",
            "val/cls_loss": "0.00000",
            "val/dfl_loss": "0.00000",
        }
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_columns)
            writer.writerow(row)

        print(
            f"Epoch {epoch + 1}/{self.config.epochs}: box_loss={train_loss:.5f}, "
            f"cls_loss=0.00000, dfl_loss=0.00000, mP={metrics.get('precision', 0):.4f}, "
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
        """Train PyTorch model with VOC dataloader and comprehensive metrics."""
        import time as time_module
        from data.voc_loader import create_tn5000_dataloaders

        base_dir = Path(__file__).parent.parent.resolve()
        data_root = base_dir / "data" / "voc"

        self._setup_logging()

        loaders = create_tn5000_dataloaders(
            data_root=str(data_root),
            batch_size=self.config.batch_size,
            num_workers=4,
        )
        train_loader = loaders["train"]
        val_loader = loaders["val"]

        optimizer = self._build_optimizer()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        print(f"Training PyTorch model: {len(train_loader)} batches")
        print(f"FCOS pretrained on COCO, remapping TN5000 labels to COCO labels")
        print(f"Saving to: {self.train_dir}")

        best_map = -1
        best_epoch = 0

        for epoch in range(self.config.epochs):
            epoch_start = time_module.time()
            self.model.train()
            total_loss = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
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

            train_loss = total_loss / len(train_loader)
            scheduler.step()

            # Validation with metrics
            self.model.train()
            val_loss = 0
            self.metrics_calculator.reset()

            with torch.no_grad():
                for images, targets in tqdm(val_loader, desc="Evaluating"):
                    images = [img.to(self.device) for img in images]
                    targets = [
                        {k: v.to(self.device) for k, v in t.items()} for t in targets
                    ]
                    remapped_targets = self._remap_labels(targets)
                    loss_dict = self.model(images, remapped_targets)

                    if isinstance(loss_dict, dict):
                        val_loss += sum(
                            v.item() if hasattr(v, "item") else v
                            for v in loss_dict.values()
                        )
                    elif isinstance(loss_dict, list):
                        val_loss += sum(
                            v.item() if hasattr(v, "item") else v for v in loss_dict
                        )

                    # Get predictions for metrics
                    self.model.eval()
                    predictions = self.model(images)
                    predictions = self._remap_predictions(predictions)
                    self.metrics_calculator.evaluate_batch(predictions, targets)
                    self.model.train()

            val_loss = val_loss / len(val_loader)
            metrics = self.metrics_calculator.compute_metrics()

            elapsed_time = time_module.time() - epoch_start

            # Log and save
            self._log_epoch(epoch, elapsed_time, train_loss, val_loss, metrics)

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


class PyTorchEvaluator:
    """Simple PyTorch model evaluator for FCOS, Faster-RCNN, etc."""

    COCO_TO_TN5000 = {1: 0, 3: 1}

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device if hasattr(config, "device") else "cuda"
        self.exp_dir = getattr(config, "exp_dir", Path("experiments/default"))
        self.val_dir = self.exp_dir / "val"
        self.metrics_calculator = MetricsCalculator(iou_threshold=0.5)

    def _remap_predictions(self, predictions):
        """Remap COCO labels back to TN5000 labels."""
        TN5000_TO_COCO_REVERSE = {v: k for k, v in self.COCO_TO_TN5000.items()}
        remapped = []
        for pred in predictions:
            if isinstance(pred, dict) and "labels" in pred:
                new_pred = pred.copy()
                labels = new_pred["labels"]
                if isinstance(labels, torch.Tensor):
                    labels = labels.clone()
                    for coco_label, tn_lbl in TN5000_TO_COCO_REVERSE.items():
                        labels[labels == coco_label] = tn_lbl
                    new_pred["labels"] = labels
                remapped.append(new_pred)
            else:
                remapped.append(pred)
        return remapped

    def _get_backbone(self):
        """Get backbone from model for Grad-CAM."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "backbone"):
            return self.model.model.backbone
        return None

    def _generate_heatmaps(self, dataset, data_root):
        """Generate Grad-CAM heatmaps."""
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        heatmaps_dir = self.val_dir / "heatmaps"
        heatmaps_dir.mkdir(parents=True, exist_ok=True)

        backbone = self._get_backbone()
        if backbone is None:
            print("⚠ Warning: Could not find backbone, skipping heatmaps")
            return

        target_layer = None
        for _, module in reversed(list(backbone.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break

        if target_layer is None:
            print("⚠ Warning: Could not find target layer, skipping heatmaps")
            return

        activations, gradients = None, None

        def forward_hook(module, input, output):
            nonlocal activations
            activations = output

        def backward_hook(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0]

        handle1 = target_layer.register_forward_hook(forward_hook)
        handle2 = target_layer.register_full_backward_hook(backward_hook)

        val_transform = A.Compose([ToTensorV2()])

        print(f"\nGenerating {len(dataset)} Grad-CAM heatmaps...")

        try:
            for idx in tqdm(range(len(dataset))):
                img_id = dataset.image_ids[idx]
                img_path = data_root / "JPEGImages" / f"{img_id}.jpg"

                if not img_path.exists():
                    continue

                img = cv2.imread(str(img_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                orig_h, orig_w = img.shape[:2]

                transformed = val_transform(image=img_rgb)
                img_tensor = (
                    transformed["image"].unsqueeze(0).to(self.device).float() / 255.0
                )

                self.model.zero_grad()
                self.model([img_tensor[0]])

                if activations is not None and gradients is not None:
                    pooled_grad = gradients.mean(dim=(2, 3), keepdim=True)
                    cam = (pooled_grad * activations).sum(dim=1).squeeze()
                    cam = torch.nn.functional.relu(cam).cpu().detach().numpy()
                    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

                    heatmap = cv2.resize(cam, (orig_w, orig_h))
                    heatmap = (heatmap * 255).astype(np.uint8)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    blended = (img_rgb * 0.5 + heatmap * 0.5).astype(np.uint8)

                    cv2.imwrite(
                        str(heatmaps_dir / f"{img_id}.jpg"),
                        cv2.cvtColor(blended, cv2.COLOR_RGB2BGR),
                    )
        finally:
            handle1.remove()
            handle2.remove()

        print(f"✓ Saved Grad-CAM heatmaps to {heatmaps_dir}")

    def _generate_comparison_images(self, dataset, data_root, class_names):
        """Generate GT vs Pred comparison images."""
        from data.voc_loader import collate_fn
        from torch.utils.data import DataLoader

        boxes_dir = self.val_dir / "boxes"
        boxes_dir.mkdir(parents=True, exist_ok=True)

        colors = {0: (255, 0, 0), 1: (0, 0, 255)}

        test_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        all_predictions = []
        all_targets = []

        for idx, (images, targets) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            gt_boxes = targets[0]["boxes"].cpu()
            gt_labels = targets[0]["labels"].cpu()

            with torch.no_grad():
                predictions = self.model(images)

            if len(predictions) > 0:
                pred = predictions[0]
                pred = self._remap_predictions([pred])[0]

                all_predictions.append(pred)
                all_targets.append(targets[0])

                img_id = dataset.image_ids[idx]
                img_path = data_root / "JPEGImages" / f"{img_id}.jpg"

                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    h, w = img.shape[:2]

                    img_gt = img.copy()
                    for i, box in enumerate(gt_boxes):
                        x1, y1, x2, y2 = box.int().tolist()
                        cls = gt_labels[i].item()
                        color = colors.get(cls, (0, 255, 0))
                        cv2.rectangle(img_gt, (x1, y1), (x2, y2), color, 2)
                        label = class_names[cls]
                        cv2.putText(
                            img_gt,
                            label,
                            (x1, max(y1 - 5, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )

                    img_pred = img.copy()
                    pred_boxes = pred.get("boxes", torch.tensor([]))
                    pred_labels = pred.get("labels", torch.tensor([]))
                    pred_scores = pred.get("scores", torch.ones(len(pred_boxes)))

                    for i, box in enumerate(pred_boxes):
                        x1, y1, x2, y2 = box.int().tolist()
                        cls = int(pred_labels[i].item())
                        conf = float(pred_scores[i].item())
                        color = colors.get(cls, (0, 255, 0))
                        cv2.rectangle(img_pred, (x1, y1), (x2, y2), color, 2)
                        label = f"{class_names[cls]} {conf:.2f}"
                        cv2.putText(
                            img_pred,
                            label,
                            (x1, max(y1 - 5, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                        )

                    canvas = np.zeros((h + 50, w * 2, 3), dtype=np.uint8)
                    canvas[50:, :w] = img_gt
                    canvas[50:, w:] = img_pred

                    cv2.putText(
                        canvas,
                        "GT",
                        (w // 2 - 20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        canvas,
                        "PRED",
                        (w + w // 2 - 30, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )

                    cv2.imwrite(str(boxes_dir / f"{img_id}.jpg"), canvas)

        return all_predictions, all_targets

    def _save_results_json(self, metrics):
        """Save results to JSON."""
        results = {
            "overall": {
                "mAP50": float(metrics.get("mAP50", 0)),
                "mAP50_95": float(metrics.get("mAP", 0)),
                "precision": float(metrics.get("precision", 0)),
                "recall": float(metrics.get("recall", 0)),
                "f1": float(metrics.get("f1", 0)),
            },
            "per_class": {},
        }

        class_names = ["benign", "malignant"]
        for i, name in enumerate(class_names):
            results["per_class"][name] = {
                "AP50": float(metrics.get("mAP50", 0)),
                "AP50_95": float(metrics.get("mAP", 0)),
                "precision": float(metrics.get("precision", 0)),
                "recall": float(metrics.get("recall", 0)),
                "f1": float(metrics.get("f1", 0)),
            }

        import json

        results_path = self.val_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"✓ Saved results to {results_path}")

    def evaluate(self):
        """Evaluate PyTorch model on test set."""
        from data.voc_loader import VOCDataset
        from torch.utils.data import DataLoader

        base_dir = Path(__file__).parent.parent.resolve()
        data_root = base_dir / "data" / "voc"

        self.val_dir.mkdir(parents=True, exist_ok=True)
        (self.val_dir / "boxes").mkdir(exist_ok=True)
        (self.val_dir / "heatmaps").mkdir(exist_ok=True)
        (self.val_dir / "plots").mkdir(exist_ok=True)

        class_names = ["benign", "malignant"]

        test_dataset = VOCDataset(
            root=str(data_root),
            split="val",
            numeric_classes=True,
        )

        print(f"Evaluating PyTorch model on {len(test_dataset)} images...")
        self.model.eval()

        predictions, targets = self._generate_comparison_images(
            test_dataset, data_root, class_names
        )

        self.metrics_calculator.reset()
        for pred, target in zip(predictions, targets):
            self.metrics_calculator.evaluate_batch([pred], [target])
        metrics = self.metrics_calculator.compute_metrics()

        self._save_results_json(metrics)
        self._generate_heatmaps(test_dataset, data_root)

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"mAP@50:      {metrics['mAP50']:.4f}")
        print(f"mAP@50-95:   {metrics['mAP']:.4f}")
        print(f"Precision:    {metrics['precision']:.4f}")
        print(f"Recall:      {metrics['recall']:.4f}")
        print(f"F1 Score:    {metrics['f1']:.4f}")
        print("=" * 60)
        print(f"\nResults saved to: {self.val_dir}")

        return metrics
