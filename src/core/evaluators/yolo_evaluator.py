import os
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

        metrics_dict = {
            "map50": round(float(metrics.box.map50), 4),
            "map50_95": round(float(metrics.box.map), 4),
            "mean_precision": round(float(metrics.box.mp), 4),
            "mean_recall": round(float(metrics.box.mr), 4),
            "f1_score": round(float(np.mean(metrics.box.f1)), 4)
            if len(metrics.box.f1) > 0
            else 0.0,
        }

        confusion_matrix = None
        if hasattr(metrics, "confusion_matrix") and hasattr(
            metrics.confusion_matrix, "matrix"
        ):
            confusion_matrix = metrics.confusion_matrix.matrix

        self._last_metrics = metrics_dict

        self.save_results(
            metrics_dict,
            extra_data={"confusion_matrix": confusion_matrix}
            if confusion_matrix is not None
            else None,
        )

        self.generate_comparison_images()
        self.generate_heatmaps()

        return metrics_dict

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

        test_dir = Path(data_info["test"])
        save_path = self.exp_dir / "boxes"
        save_path.mkdir(parents=True, exist_ok=True)

        colors = {0: (255, 0, 0), 1: (0, 0, 255)}

        img_files = list(test_dir.glob("*.[jp][pn]g"))

        for img_path in img_files:
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
                cv2.rectangle(
                    img_pred,
                    (coords[0], coords[1]),
                    (coords[2], coords[3]),
                    colors.get(cls, (0, 255, 0)),
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

        img_files = list(Path(data_info["test"]).glob("*.[jp][pn]g"))
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

        for img_path in img_files[:10]:
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
