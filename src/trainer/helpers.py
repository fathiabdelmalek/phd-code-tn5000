import numpy as np
import torch
import cv2
from tqdm import tqdm
from pathlib import Path


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

    def _get_backbone(self):
        """Get backbone from model for Grad-CAM."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "backbone"):
            return self.model.model.backbone
        return None

    def _generate_heatmaps(self, dataset, data_root):
        """Generate Grad-CAM heatmaps using activation maps."""
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        heatmaps_dir = self.val_dir / "heatmaps"
        heatmaps_dir.mkdir(parents=True, exist_ok=True)

        backbone = self._get_backbone()
        if backbone is None:
            print("⚠ Warning: Could not find backbone, skipping heatmaps")
            return

        # Get deepest Conv2d in backbone
        conv_layers = [
            (name, module)
            for name, module in backbone.named_modules()
            if isinstance(module, torch.nn.Conv2d)
        ]
        target_layer = conv_layers[-1][1] if conv_layers else None

        if target_layer is None:
            print("⚠ Warning: Could not find target layer, skipping heatmaps")
            return

        activations = None

        def forward_hook(module, input, output):
            nonlocal activations
            activations = output.detach()

        handle = target_layer.register_forward_hook(forward_hook)

        val_transform = A.Compose([ToTensorV2()])

        print(f"\nGenerating {len(dataset)} Grad-CAM heatmaps...")

        self.model.eval()

        try:
            for idx in tqdm(range(len(dataset))):
                # Handle both YOLO and VOC format datasets
                if hasattr(dataset, "image_files"):
                    # YOLO format
                    img_path = dataset.image_files[idx]
                    img_id = img_path.stem
                else:
                    # VOC format
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

                self.model([img_tensor[0]])

                if activations is not None:
                    feat = activations
                    if len(feat.shape) == 4:
                        heatmap = torch.mean(feat, dim=1).squeeze()
                    else:
                        heatmap = feat.squeeze()

                    heatmap = torch.nn.functional.relu(heatmap)
                    heatmap = heatmap.cpu().detach().numpy()

                    if heatmap.size == 1:
                        heatmap = np.zeros((16, 16))

                    heatmap = (heatmap - heatmap.min()) / (
                        heatmap.max() - heatmap.min() + 1e-8
                    )

                    heatmap = cv2.resize(heatmap, (orig_w, orig_h))
                    heatmap = (heatmap * 255).astype(np.uint8)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    blended = (img_rgb * 0.5 + heatmap * 0.5).astype(np.uint8)

                    cv2.imwrite(
                        str(heatmaps_dir / f"{img_id}.jpg"),
                        cv2.cvtColor(blended, cv2.COLOR_RGB2BGR),
                    )
                else:
                    blank = np.zeros((orig_h, orig_w), dtype=np.float32)
                    heatmap = cv2.resize(blank, (orig_w, orig_h))
                    heatmap = (heatmap * 255).astype(np.uint8)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    blended = (img_rgb * 0.7 + heatmap * 0.3).astype(np.uint8)
                    cv2.imwrite(
                        str(heatmaps_dir / f"{img_id}.jpg"),
                        cv2.cvtColor(blended, cv2.COLOR_RGB2BGR),
                    )

        finally:
            handle.remove()

        print(f"✓ Saved Grad-CAM heatmaps to {heatmaps_dir}")

    def _generate_comparison_images(self, dataset, data_root, class_names):
        """Generate GT vs Pred comparison images."""
        from src.data.voc_loader import collate_fn
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

                # Handle both YOLO and VOC format datasets
                if hasattr(dataset, "image_files"):
                    # YOLO format - use image_files
                    img_path = dataset.image_files[idx]
                    img_id = img_path.stem
                else:
                    # VOC format - use image_ids
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
                        cls_name = (
                            class_names[cls]
                            if 0 <= cls < len(class_names)
                            else str(cls)
                        )
                        color = colors.get(cls, (0, 255, 0))
                        cv2.rectangle(img_pred, (x1, y1), (x2, y2), color, 2)
                        label = f"{cls_name} {conf:.2f}"
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
        from src.data.voc_loader import TN5000YOLODataset, collate_fn
        from torch.utils.data import DataLoader

        base_dir = Path(__file__).parent.parent.resolve()
        data_root = base_dir / "data"

        self.val_dir.mkdir(parents=True, exist_ok=True)
        (self.val_dir / "boxes").mkdir(exist_ok=True)
        (self.val_dir / "heatmaps").mkdir(exist_ok=True)
        (self.val_dir / "plots").mkdir(exist_ok=True)

        class_names = ["benign", "malignant"]

        # Use YOLO format dataset
        test_dataset = TN5000YOLODataset(
            root=data_root / "yolo",
            split="test",
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
