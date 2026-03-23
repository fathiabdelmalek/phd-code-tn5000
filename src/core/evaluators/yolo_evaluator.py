import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

from ..base_evaluator import BaseEvaluator
from ...datasets.factory import get_dataloader


class YOLOEvaluator(BaseEvaluator):
    """Evaluator implementation for YOLO models."""

    def __init__(
        self,
        model,
        data_cfg: str,
        exp_dir: str,
        class_names: Optional[List[str]] = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        imgsz: int = 640,
    ):
        super().__init__(
            model=model,
            dataloader=data_cfg,
            exp_dir=exp_dir,
            class_names=class_names or ["benign", "malignant"],
            conf_thres=conf_thres,
            iou_thres=iou_thres,
        )
        self.data_cfg = data_cfg
        self.imgsz = imgsz

    def evaluate(self) -> Dict[str, float]:
        """Run full evaluation on the test set."""
        abs_exp_dir = os.path.abspath(self.exp_dir)
        val_dir_name = os.path.join(os.path.basename(abs_exp_dir), "val")

        metrics = self.model.val(
            data=self.data_cfg,
            imgsz=self.imgsz,
            split="test",
            conf=self.conf_thres,
            iou=self.iou_thres,
            plots=True,
            project=os.path.dirname(abs_exp_dir),
            name=val_dir_name,
        )

        # Build comprehensive metrics dictionary
        metrics_dict = self._build_comprehensive_metrics(metrics)

        confusion_matrix = None
        if hasattr(metrics, "confusion_matrix") and hasattr(
            metrics.confusion_matrix, "matrix"
        ):
            confusion_matrix = (
                metrics.confusion_matrix.matrix.tolist()
                if hasattr(metrics.confusion_matrix.matrix, "tolist")
                else metrics.confusion_matrix.matrix
            )

        self._last_metrics = metrics_dict

        # Save comprehensive metrics to JSON
        self._save_metrics_json(metrics_dict, confusion_matrix)

        self.save_results(
            metrics_dict,
            extra_data={"confusion_matrix": confusion_matrix}
            if confusion_matrix is not None
            else None,
        )

        self.generate_comparison_images()
        self.generate_heatmaps()
        self._save_plots(abs_exp_dir)

        return metrics_dict

    def _build_comprehensive_metrics(self, metrics) -> Dict[str, Any]:
        """Build a comprehensive metrics dictionary with per-class metrics."""
        box = metrics.box
        class_names = self.class_names

        # Get per-class metrics
        ap50 = box.ap50 if hasattr(box, "ap50") else []
        ap50_95 = box.ap if hasattr(box, "ap") else []
        precision = box.p if hasattr(box, "p") else []
        recall = box.r if hasattr(box, "r") else []
        f1 = box.f1 if hasattr(box, "f1") else []

        # Build per-class metrics
        per_class_metrics = {}
        for i, name in enumerate(class_names):
            per_class_metrics[name] = {
                "AP50": round(float(ap50[i]), 4) if i < len(ap50) else 0.0,
                "AP50_95": round(float(ap50_95[i]), 4) if i < len(ap50_95) else 0.0,
                "precision": round(float(precision[i]), 4)
                if i < len(precision)
                else 0.0,
                "recall": round(float(recall[i]), 4) if i < len(recall) else 0.0,
                "f1_score": round(float(f1[i]), 4) if i < len(f1) else 0.0,
            }

        # Build overall metrics
        metrics_dict = {
            "overall": {
                "mAP50": round(float(box.map50), 4),
                "mAP50_95": round(float(box.map), 4),
                "mean_precision": round(float(box.mp), 4),
                "mean_recall": round(float(box.mr), 4),
                "mean_f1_score": round(float(np.mean(box.f1)), 4)
                if len(box.f1) > 0
                else 0.0,
            },
            "per_class": per_class_metrics,
            "confusion_matrix": None,  # Will be filled separately
            "config": {
                "conf_thres": self.conf_thres,
                "iou_thres": self.iou_thres,
                "imgsz": self.imgsz,
            },
        }

        return metrics_dict

    def _save_metrics_json(self, metrics_dict: Dict, confusion_matrix: Any) -> None:
        """Save comprehensive metrics to JSON file."""
        # Update confusion matrix if available
        if confusion_matrix is not None:
            metrics_dict["confusion_matrix"] = confusion_matrix

        # Save to results directory
        results_dir = Path(self.exp_dir) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        json_path = results_dir / "metrics.json"
        with open(json_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)

        print(f"Metrics saved to: {json_path}")

    def _save_plots(self, exp_dir: str) -> None:
        """Copy important plots to the plots directory."""
        import shutil

        exp_path = Path(exp_dir)
        plots_dir = exp_path / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Plots to copy
        plot_files = [
            "BoxF1_curve.png",
            "BoxP_curve.png",
            "BoxPR_curve.png",
            "BoxR_curve.png",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "results.png",
        ]

        copied = []
        for plot_name in plot_files:
            src = exp_path / plot_name
            if src.exists():
                dst = plots_dir / plot_name
                shutil.copy2(src, dst)
                copied.append(plot_name)

        if copied:
            print(f"Copied {len(copied)} plots to: {plots_dir}")

    def predict(self, *args, **kwargs):
        """Run inference."""
        return self.model(*args, **kwargs)

    def get_metrics(self) -> Dict[str, float]:
        """Return the most recent evaluation metrics."""
        return self._last_metrics

    def generate_comparison_images(self) -> None:
        """Generate ground truth vs prediction comparison images."""
        import cv2
        import yaml

        with open(self.data_cfg, "r") as f:
            data_info = yaml.safe_load(f)

        # Resolve test path relative to the yaml's path key
        yaml_dir = Path(self.data_cfg).parent
        data_root = data_info.get("path", str(yaml_dir))
        test_dir = (
            yaml_dir / data_info["test"]
            if not os.path.isabs(data_info["test"])
            else Path(data_info["test"])
        )

        save_path = self.exp_dir / "boxes"
        save_path.mkdir(parents=True, exist_ok=True)

        colors = {0: (255, 0, 0), 1: (0, 0, 255)}

        img_files = list(test_dir.glob("*.[jp][pn]g"))

        print(f"Generating comparison images for {len(img_files)} images...")

        for i, img_path in enumerate(img_files):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]

            img_gt = img.copy()
            lbl_path = (
                img_path.as_posix()
                .replace("/images/", "/labels/")
                .replace(".jpg", ".txt")
                .replace(".png", ".txt")
            )
            if os.path.exists(lbl_path):
                for line in open(lbl_path):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, x, y, nw, nh = map(float, parts)
                        cv2.rectangle(
                            img_gt,
                            (int((x - nw / 2) * w), int((y - nh / 2) * h)),
                            (int((x + nw / 2) * w), int((y + nh / 2) * h)),
                            colors[int(cls)],
                            2,
                        )

            img_pred = img.copy()
            for b in self.model(str(img_path), verbose=False, conf=self.conf_thres)[
                0
            ].boxes:
                coords = b.xyxy[0].cpu().numpy().astype(int)
                cls = int(b.cls)
                conf = float(b.conf[0]) if hasattr(b, "conf") else 1.0
                color = colors.get(cls, (0, 255, 0))
                cv2.rectangle(
                    img_pred,
                    (coords[0], coords[1]),
                    (coords[2], coords[3]),
                    color,
                    2,
                )
                label = f"{conf:.2f}"
                cv2.putText(
                    img_pred,
                    label,
                    (coords[0], max(coords[1] - 5, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            canvas = np.zeros((h + 50, w * 2, 3), np.uint8)
            canvas[50:, :w] = img_gt
            canvas[50:, w:] = img_pred
            cv2.putText(canvas, "GT", (w // 2 - 20, 35), 0, 0.8, (255, 255, 255), 2)
            cv2.putText(
                canvas, "PRED", (w + w // 2 - 30, 35), 0, 0.8, (255, 255, 255), 2
            )

            cv2.imwrite(str(save_path / img_path.name), canvas)

    def generate_heatmaps(self) -> None:
        """Generate activation heatmaps for feature visualization."""
        import cv2
        import torch
        import yaml

        with open(self.data_cfg, "r") as f:
            data_info = yaml.safe_load(f)

        # Resolve test path relative to the yaml's path key
        yaml_dir = Path(self.data_cfg).parent
        data_root = data_info.get("path", str(yaml_dir))
        test_dir = (
            yaml_dir / data_info["test"]
            if not os.path.isabs(data_info["test"])
            else Path(data_info["test"])
        )

        img_files = list(test_dir.glob("*.[jp][pn]g"))
        save_path = self.exp_dir / "heatmaps"
        save_path.mkdir(parents=True, exist_ok=True)

        activations = {}

        def hook_fn(m, i, o):
            activations["feat"] = o

        try:
            backbone = list(self.model.model.model.children())
            target_layer = backbone[15] if len(backbone) > 15 else backbone[-2]
            handle = target_layer.register_forward_hook(hook_fn)
        except Exception:
            return

        print(f"Generating heatmaps for {len(img_files)} images...")
        for i, img_path in enumerate(img_files):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            resized = cv2.resize(img, (self.imgsz, self.imgsz))
            tensor = (
                torch.from_numpy(resized.transpose(2, 0, 1))
                .float()
                .div(255)
                .unsqueeze(0)
                .to(self.model.device)
            )

            with torch.no_grad():
                _ = self.model(tensor)

            if "feat" in activations:
                feat = activations["feat"].squeeze(0).mean(dim=0).cpu().numpy()
                feat = cv2.resize(np.maximum(feat, 0), (img.shape[1], img.shape[0]))
                feat = (feat / (feat.max() + 1e-8) * 255).astype(np.uint8)

                heatmap = cv2.applyColorMap(feat, cv2.COLORMAP_JET)
                result = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
                cv2.imwrite(str(save_path / img_path.name), result)

        handle.remove()
