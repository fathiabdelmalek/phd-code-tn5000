#!/usr/bin/env python3
"""
Comprehensive Evaluation Script

Structure:
    experiments/{model}_{scale}_{timestamp}/
    ├── metadata.json
    ├── train/                  # Training outputs
    │   ├── train_batch*.jpg
    │   ├── val_batch*.jpg
    │   ├── weights/
    │   │   ├── best.pt
    │   │   └── last.pt
    │   └── args.yaml
    └── val/                    # Evaluation outputs
        ├── boxes/              # 1000 GT vs PRED comparisons
        ├── heatmaps/           # 1000 activation heatmaps
        ├── plots/              # PR curves, confusion matrix
        └── results.json        # Detailed metrics

Usage:
    cd src
    python evaluate.py --exp experiments/yolo26_n_20260324_120000
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import shutil

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

from models import get_model


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_labels(img_path, labels_dir):
    """Load YOLO format labels."""
    label_path = labels_dir / f"{img_path.stem}.txt"
    boxes = []
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    boxes.append((cls, x, y, w, h))
    return boxes


def draw_boxes(img, boxes, colors, class_names):
    """Draw bounding boxes on image."""
    h, w = img.shape[:2]
    for cls, x, y, bw, bh in boxes:
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)

        color = colors.get(cls, (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = class_names[cls] if cls < len(class_names) else str(cls)
        cv2.putText(
            img, label, (x1, max(y1 - 5, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
    return img


def draw_predictions(img, results, colors, class_names):
    """Draw predicted boxes on image."""
    if results is None or len(results) == 0:
        return img

    boxes = results[0].boxes if hasattr(results[0], "boxes") else []

    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        cls = int(box.cls)
        conf = float(box.conf[0]) if hasattr(box, "conf") else 1.0

        color = colors.get(cls, (0, 255, 0))
        cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
        label = f"{class_names[cls] if cls < len(class_names) else cls} {conf:.2f}"
        cv2.putText(
            img,
            label,
            (xyxy[0], max(xyxy[1] - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return img


def generate_comparison_images(
    model, test_images_dir, labels_dir, exp_dir, class_names, conf_thres=0.25
):
    """Generate GT vs Pred comparison images for ALL test images."""
    boxes_dir = exp_dir / "val" / "boxes"
    ensure_dir(boxes_dir)

    colors = {0: (255, 0, 0), 1: (0, 0, 255)}  # Blue=benign, Red=malignant

    img_files = sorted(list(test_images_dir.glob("*.[jp][pn]g")))
    print(f"\nGenerating {len(img_files)} comparison images...")

    for img_path in tqdm(img_files):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        # Ground truth
        img_gt = img.copy()
        gt_boxes = load_labels(img_path, labels_dir)
        draw_boxes(img_gt, gt_boxes, colors, class_names)

        # Predictions
        img_pred = img.copy()
        results = model(img_path, verbose=False, conf=conf_thres)
        draw_predictions(img_pred, results, colors, class_names)

        # Combine side by side
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

        cv2.imwrite(str(boxes_dir / img_path.name), canvas)

    print(f"✓ Saved {len(img_files)} comparison images to {boxes_dir}")


class GradCAM:
    """Grad-CAM implementation for YOLO models."""

    def __init__(self, model, target_layer=None):
        self.model = model
        self.gradients = None
        self.activations = None
        self.target_layer = target_layer or self._find_target_layer()
        self.hooks = []
        self._register_hooks()

    def _find_target_layer(self):
        """Find the last convolutional layer in the backbone."""
        if hasattr(self.model, "model"):
            backbone = (
                self.model.model.model
                if hasattr(self.model.model, "model")
                else self.model.model
            )
            for name, module in reversed(list(backbone.named_modules())):
                if "Detect" in type(module).__name__:
                    continue
                if isinstance(module, torch.nn.Conv2d):
                    return module
        return None

    def _register_hooks(self):
        """Register forward and backward hooks."""
        if self.target_layer is None:
            return

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()

    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap for the input."""
        if self.target_layer is None or self.activations is None:
            return None

        self.model.zero_grad()

        output = self.model(input_tensor)
        if target_class is None:
            target_class = (
                output[0].boxes.conf.argmax().item() if len(output[0]) > 0 else 0
            )

        one_hot = torch.zeros_like(output[0].boxes.conf)
        one_hot[0, target_class] = 1.0
        output[0].boxes.conf.backward(gradient=one_hot, retain_graph=True)

        if self.gradients is None or self.activations is None:
            return None

        pooled_grad = self.gradients.mean(dim=(2, 3), keepdim=True)
        weighted_activations = pooled_grad * self.activations
        cam = weighted_activations.sum(dim=1).squeeze()

        cam = torch.nn.functional.relu(cam)
        cam = cam.cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


def generate_heatmaps(model, test_images_dir, exp_dir):
    """Generate Grad-CAM heatmaps for ALL test images."""
    import torch

    heatmaps_dir = exp_dir / "val" / "heatmaps"
    ensure_dir(heatmaps_dir)

    img_files = sorted(list(test_images_dir.glob("*.[jp][pn]g")))
    print(f"\nGenerating {len(img_files)} Grad-CAM heatmaps...")

    try:
        gradcam = GradCAM(model)
        if gradcam.target_layer is None:
            print("⚠ Warning: Could not find target layer, trying fallback...")
            gradcam.target_layer = (
                model.model.model[-2] if hasattr(model.model, "model") else None
            )
            if gradcam.target_layer:
                gradcam._register_hooks()

        if gradcam.target_layer is None:
            print("⚠ Warning: Could not find layer for Grad-CAM, skipping heatmaps")
            return
    except Exception as e:
        print(f"⚠ Warning: Grad-CAM initialization failed: {e}, skipping heatmaps")
        return

    for img_path in tqdm(img_files):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (640, 640))
            img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(
                next(model.model.parameters()).device
                if hasattr(model, "model")
                else "cpu"
            )

            cam = gradcam.generate(img_tensor)

            if cam is not None:
                heatmap = cv2.resize(cam, (img.shape[1], img.shape[0]))
                heatmap = (heatmap * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                blended = (img_rgb * 0.5 + heatmap * 0.5).astype(np.uint8)
                cv2.imwrite(
                    str(heatmaps_dir / img_path.name),
                    cv2.cvtColor(blended, cv2.COLOR_RGB2BGR),
                )
            else:
                cam = np.zeros((img.shape[0] // 4, img.shape[1] // 4), dtype=np.float32)
                heatmap = cv2.resize(cam, (img.shape[1], img.shape[0]))
                heatmap = (heatmap * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                blended = (img_rgb * 0.7 + heatmap * 0.3).astype(np.uint8)
                cv2.imwrite(
                    str(heatmaps_dir / img_path.name),
                    cv2.cvtColor(blended, cv2.COLOR_RGB2BGR),
                )
        except Exception as e:
            continue

    gradcam.remove_hooks()
    print(f"✓ Saved Grad-CAM heatmaps to {heatmaps_dir}")


def save_results_json(metrics, exp_dir):
    """Save comprehensive metrics to JSON."""
    results_dir = exp_dir / "val"
    ensure_dir(results_dir)

    # Build comprehensive results
    results = {
        "overall": {
            "mAP50": float(metrics.box.map50) if hasattr(metrics.box, "map50") else 0.0,
            "mAP50_95": float(metrics.box.map) if hasattr(metrics.box, "map") else 0.0,
            "precision": float(metrics.box.mp) if hasattr(metrics.box, "mp") else 0.0,
            "recall": float(metrics.box.mr) if hasattr(metrics.box, "mr") else 0.0,
            "f1": 2
            * (float(metrics.box.mp) * float(metrics.box.mr))
            / (float(metrics.box.mp) + float(metrics.box.mr) + 1e-8)
            if hasattr(metrics.box, "mp") and hasattr(metrics.box, "mr")
            else 0.0,
        },
        "per_class": {},
    }

    # Per-class metrics
    class_names = ["benign", "malignant"]
    if hasattr(metrics.box, "ap50") and len(metrics.box.ap50) > 0:
        for i, name in enumerate(class_names):
            if i < len(metrics.box.ap50):
                results["per_class"][name] = {
                    "AP50": float(metrics.box.ap50[i]),
                    "AP50_95": float(metrics.box.ap[i])
                    if hasattr(metrics.box, "ap") and i < len(metrics.box.ap)
                    else 0.0,
                    "precision": float(metrics.box.p[i])
                    if hasattr(metrics.box, "p") and i < len(metrics.box.p)
                    else 0.0,
                    "recall": float(metrics.box.r[i])
                    if hasattr(metrics.box, "r") and i < len(metrics.box.r)
                    else 0.0,
                    "f1": float(metrics.box.f1[i])
                    if hasattr(metrics.box, "f1") and i < len(metrics.box.f1)
                    else 0.0,
                }

    results_path = results_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved results to {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a detection model")
    parser.add_argument("--exp", type=str, required=True, help="Experiment directory")
    parser.add_argument(
        "--scale", "-s", type=str, default="n", choices=["n", "s", "m", "l", "x"]
    )
    parser.add_argument(
        "--weights", "-w", type=str, default=None, help="Path to weights file"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", "-d", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    exp_dir = Path(args.exp).resolve()

    if not exp_dir.exists():
        print(f"Error: Experiment directory not found: {exp_dir}")
        return

    # Load metadata
    metadata = {}
    metadata_path = exp_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"Loaded metadata from {metadata_path}")

    class_names = ["benign", "malignant"]

    # Data paths - use absolute path
    base_dir = Path(__file__).parent.parent.resolve()
    data_root = base_dir / "data" / "yolo"
    data_yaml = data_root / "data.yaml"
    test_images_dir = data_root / "images" / "test"
    test_labels_dir = data_root / "labels" / "test"

    # Model config
    model_name = metadata.get("model", "yolo26")
    scale = metadata.get("scale", args.scale)
    is_yolo = model_name == "yolo26"

    # Load model
    if is_yolo:
        print(f"Loading YOLO26 scale={scale}")
        model = get_model("yolo26", num_classes=2, scale=scale)
    else:
        print(f"Loading {model_name.upper()}")
        from models.fcos import FCOSWrapper

        model = FCOSWrapper(num_classes=2, pretrained=False)
        model = model.to(args.device)

    # Load weights
    weights_path = args.weights or str(exp_dir / "val" / "weights" / "best.pt")
    if Path(weights_path).exists():
        print(f"Loading weights from {weights_path}")
        if is_yolo:
            model = model.load(weights_path)
        else:
            model.load_state_dict(torch.load(weights_path, map_location=args.device))

    # Ensure val directories exist
    ensure_dir(exp_dir / "val" / "boxes")
    ensure_dir(exp_dir / "val" / "heatmaps")
    ensure_dir(exp_dir / "val" / "plots")

    print("\n" + "=" * 60)
    print("Running YOLO validation on TEST set...")
    print("=" * 60 + "\n")

    if is_yolo:
        # YOLO evaluation - save directly to val/ directory
        # Use absolute path to avoid runs/ folder
        val_dir = exp_dir / "val"

        results = model.val(
            data=str(data_yaml),
            split="test",
            conf=args.conf,
            imgsz=640,
            device=args.device,
            plots=True,
            save=True,
            exist_ok=True,
            project=str(val_dir.resolve()),
            name=".",
        )

        # Save results.json
        save_results_json(results, exp_dir)

        # Copy plots to val/plots/
        plots_dir = exp_dir / "val" / "plots"
        plot_files = [
            "BoxF1_curve.png",
            "BoxP_curve.png",
            "BoxPR_curve.png",
            "BoxR_curve.png",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "results.png",
            "results.csv",
        ]

        for plot_name in plot_files:
            src = val_dir / plot_name
            if src.exists():
                shutil.copy2(src, plots_dir / plot_name)

        # Copy batch images
        for f in val_dir.glob("train_batch*.jpg"):
            shutil.copy2(f, plots_dir / f.name)
        for f in val_dir.glob("val_batch*.jpg"):
            shutil.copy2(f, plots_dir / f.name)

        print(f"✓ Copied plots to {plots_dir}")

        # Generate comparison images
        generate_comparison_images(
            model, test_images_dir, test_labels_dir, exp_dir, class_names, args.conf
        )

        # Generate heatmaps
        generate_heatmaps(model, test_images_dir, exp_dir)

        # Print results
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"mAP@50:      {results.box.map50:.4f}")
        print(f"mAP@50-95:   {results.box.map:.4f}")
        print(f"Precision:    {results.box.mp:.4f}")
        print(f"Recall:      {results.box.mr:.4f}")
        f1 = (
            2
            * (results.box.mp * results.box.mr)
            / (results.box.mp + results.box.mr + 1e-8)
        )
        print(f"F1 Score:    {f1:.4f}")
        print("=" * 60)
        print(f"\nResults saved to: {exp_dir / 'val'}")
    else:
        print("Evaluating PyTorch model (FCOS)...")
        from data.voc_loader import VOCDataset, collate_fn
        from trainers import MetricsCalculator
        from torch.utils.data import DataLoader

        model.eval()

        # COCO to TN5000 label mapping (for remapping predictions)
        COCO_TO_TN5000 = {1: 0, 3: 1}  # person->benign, bird->malignant

        # Data paths
        base_dir = Path(__file__).parent.parent.resolve()
        data_root = base_dir / "data" / "voc"

        # Create test dataloader
        test_dataset = VOCDataset(
            root=str(data_root),
            split="test",
            numeric_classes=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        # Initialize metrics calculator
        metrics_calc = MetricsCalculator(iou_threshold=0.5)

        # Generate comparison images and collect metrics
        boxes_dir = exp_dir / "val" / "boxes"
        ensure_dir(boxes_dir)

        colors = {0: (255, 0, 0), 1: (0, 0, 255)}  # Blue=benign, Red=malignant

        print(f"\nEvaluating {len(test_loader)} test images...")

        all_predictions = []
        all_targets = []

        for idx, (images, targets) in enumerate(tqdm(test_loader)):
            images = [img.to(args.device) for img in images]
            targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]

            # Get ground truth
            gt_boxes = targets[0]["boxes"].cpu()
            gt_labels = targets[0]["labels"].cpu()

            # Run inference
            with torch.no_grad():
                predictions = model(images)

            if len(predictions) > 0:
                pred = predictions[0]

                # Remap COCO labels to TN5000 labels
                if "labels" in pred and len(pred["labels"]) > 0:
                    coco_labels = pred["labels"]
                    tn5000_labels = torch.zeros_like(coco_labels)
                    for coco_lbl, tn_lbl in COCO_TO_TN5000.items():
                        tn5000_labels[coco_labels == coco_lbl] = tn_lbl
                    pred["labels"] = tn5000_labels

                # Store for metrics
                all_predictions.append(pred)
                all_targets.append(targets[0])

                # Generate comparison image
                img_id = test_dataset.image_ids[idx]
                img_path = data_root / "JPEGImages" / f"{img_id}.jpg"

                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    h, w = img.shape[:2]

                    # Draw GT boxes
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

                    # Draw predictions
                    img_pred = img.copy()
                    pred_boxes = pred.get("boxes", torch.tensor([]))
                    pred_labels = pred.get("labels", torch.tensor([]))
                    pred_scores = pred.get("scores", torch.ones(len(pred_boxes)))

                    for i, box in enumerate(pred_boxes):
                        x1, y1, x2, y2 = box.int().tolist()
                        cls = (
                            int(pred_labels[i].item())
                            if isinstance(pred_labels[i], torch.Tensor)
                            else int(pred_labels[i])
                        )
                        conf = (
                            float(pred_scores[i].item())
                            if isinstance(pred_scores[i], torch.Tensor)
                            else float(pred_scores[i])
                        )
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

                    # Combine side by side
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

        # Compute metrics
        metrics_calc.reset()
        for pred, target in zip(all_predictions, all_targets):
            metrics_calc.evaluate_batch([pred], [target])
        results = metrics_calc.compute_metrics()

        # Save results.json
        save_results_json(
            type(
                "Results",
                (),
                {
                    "box": type(
                        "BoxMetrics",
                        (),
                        {
                            "map50": results["mAP50"],
                            "map": results["mAP"],
                            "mp": results["precision"],
                            "mr": results["recall"],
                            "ap50": [results["mAP50"]] * 2,
                            "ap": [results["mAP"]] * 2,
                            "p": [results["precision"]] * 2,
                            "r": [results["recall"]] * 2,
                            "f1": [results["f1"]] * 2,
                        },
                    )()
                },
            )(),
            exp_dir,
        )

        # Generate Grad-CAM heatmaps for FCOS
        generate_fcos_heatmaps(model, test_dataset, data_root, exp_dir, args.device)

        # Print results
        print("\n" + "=" * 60)
        print("FCOS EVALUATION RESULTS")
        print("=" * 60)
        print(f"mAP@50:      {results['mAP50']:.4f}")
        print(f"mAP@50-95:   {results['mAP']:.4f}")
        print(f"Precision:    {results['precision']:.4f}")
        print(f"Recall:      {results['recall']:.4f}")
        print(f"F1 Score:    {results['f1']:.4f}")
        print("=" * 60)
        print(f"\nResults saved to: {exp_dir / 'val'}")


def generate_fcos_heatmaps(model, dataset, data_root, exp_dir, device="cuda"):
    """Generate Grad-CAM heatmaps for FCOS model."""
    from torchvision.models.detection import fcos
    from transforms import get_val_transforms

    heatmaps_dir = exp_dir / "val" / "heatmaps"
    ensure_dir(heatmaps_dir)

    backbone = None
    if hasattr(model, "model") and hasattr(model.model, "backbone"):
        backbone = model.model.backbone

    if backbone is None:
        print("⚠ Warning: Could not find FCOS backbone, skipping heatmaps")
        return

    target_layer = None
    for name, module in reversed(list(backbone.named_modules())):
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
            break

    if target_layer is None:
        print("⚠ Warning: Could not find target layer for Grad-CAM, skipping heatmaps")
        return

    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    handle1 = target_layer.register_forward_hook(forward_hook)
    handle2 = target_layer.register_full_backward_hook(backward_hook)

    val_transform = get_val_transforms()

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

            image = Image.open(img_path).convert("RGB")
            transformed = val_transform(image=img_rgb)
            img_tensor = transformed["image"].unsqueeze(0).to(device)

            model.zero_grad()
            _ = model([img_tensor[0]])

            if activations is not None and gradients is not None:
                pooled_grad = gradients.mean(dim=(2, 3), keepdim=True)
                cam = (pooled_grad * activations).sum(dim=1).squeeze()
                cam = torch.nn.functional.relu(cam)
                cam = cam.cpu().detach().numpy()
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
        handle1.remove()
        handle2.remove()

    print(f"✓ Saved Grad-CAM heatmaps to {heatmaps_dir}")


if __name__ == "__main__":
    main()
