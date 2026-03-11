from abc import ABC, abstractmethod
from pathlib import Path
import os, json, yaml

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import torch


class BaseEvaluator(ABC):
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader

    @abstractmethod
    def __call__(self, data_cfg):
        """Must return a dictionary of metrics like {'map50': 0.85, 'map50-95': 0.60}"""
        pass

    def _save_results(self, exp_dir, metrics, confusion_matrix_data, train_loss=None, val_loss=None):
        """
        Standardized export for all models.
        """
        results_dir = os.path.join(exp_dir, "results")
        os.makedirs(results_dir, exist_ok=True)  # Safety first!

        # 1. Save metrics.json
        with open(os.path.join(results_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        # 2. Save Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix_data, annot=True, fmt='.0f', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(results_dir, "confusion_matrix.jpg"))
        plt.close()

        # 3. Save Loss/Training Plots (Only if data provided)
        if train_loss and val_loss:
            plt.figure()
            plt.plot(train_loss, label='Train Loss')
            plt.plot(val_loss, label='Val Loss')
            plt.legend()
            plt.title("Training Loss Curve")
            plt.savefig(os.path.join(results_dir, "loss_plot.pdf"))
            plt.close()

        print(f"✅ Results exported to {results_dir}")


class YOLOEvaluator(BaseEvaluator):
    def __call__(self, data_cfg, exp_dir):
        # 1. Standard Validation
        abs_exp_dir = os.path.abspath(exp_dir)
        val_dir_name = os.path.join(os.path.basename(abs_exp_dir), 'val')

        metrics = self.model.val(
            data=data_cfg, imgsz=640, split='test', plots=True,
            project=os.path.dirname(abs_exp_dir), name=val_dir_name,
        )

        # 2. Parallel Pipelines
        self._generate_comparison_images(data_cfg, exp_dir)
        self._generate_heatmaps(data_cfg, exp_dir)

        # 3. Compile Metrics
        metrics_dict = {
            "map50": round(metrics.box.map50, 4),
            "map50_95": round(metrics.box.map, 4),
            "mean_precision": round(metrics.box.mp, 4),
            "mean_recall": round(metrics.box.mr, 4),
            "f1_score": round(float(np.mean(metrics.box.f1)), 4),
        }
        self._save_results(exp_dir, metrics_dict, metrics.confusion_matrix.matrix)
        return metrics_dict

    def _generate_heatmaps(self, data_cfg, exp_dir):
        """Optimized heatmaps with automatic hook cleanup."""
        with open(data_cfg, 'r') as f:
            data_info = yaml.safe_load(f)

        img_files = list(Path(data_info['test']).glob('*.[jp][pn]g'))
        save_path = os.path.join(exp_dir, "heatmaps")
        os.makedirs(save_path, exist_ok=True)

        # Hook Management
        activations = {}

        def hook_fn(m, i, o): activations['feat'] = o

        # Safe access to backbone
        backbone = list(self.model.model.model.children())
        target_layer = backbone[15] if len(backbone) > 15 else backbone[-2]
        handle = target_layer.register_forward_hook(hook_fn)

        print(f"🔥 Generating heatmaps -> {save_path}")

        for img_path in img_files:
            img = cv2.imread(str(img_path))
            # Standard preprocessing
            tensor = torch.from_numpy(cv2.resize(img, (640, 640)).transpose(2, 0, 1)).float().div(255).unsqueeze(0).to(
                self.model.device)

            with torch.no_grad():
                _ = self.model(tensor)

            # Process feature map
            feat = activations['feat'].squeeze(0).mean(dim=0).cpu().numpy()
            feat = cv2.resize(np.maximum(feat, 0), (img.shape[1], img.shape[0]))
            feat = (feat / (feat.max() + 1e-8) * 255).astype(np.uint8)

            heatmap = cv2.applyColorMap(feat, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(save_path, img_path.name), cv2.addWeighted(img, 0.6, heatmap, 0.4, 0))

        handle.remove()  # Crucial: Clean up hook to prevent memory leaks

    def _generate_comparison_images(self, data_cfg, exp_dir):
        """Clean implementation of GT vs Pred visualization."""
        with open(data_cfg, 'r') as f:
            test_dir = Path(yaml.safe_load(f)['test'])

        save_path = os.path.join(exp_dir, "boxes")
        os.makedirs(save_path, exist_ok=True)
        colors = {0: (255, 0, 0), 1: (0, 0, 255)}

        for img_path in test_dir.glob('*.[jp][pn]g'):
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]

            # Draw GT (Left)
            lbl = img_path.as_posix().replace('/images/', '/labels/').replace('.jpg', '.txt').replace('.png', '.txt')
            img_gt = img.copy()
            if os.path.exists(lbl):
                for line in open(lbl):
                    cls, x, y, nw, nh = map(float, line.split())
                    cv2.rectangle(img_gt, (int((x - nw / 2) * w), int((y - nh / 2) * h)),
                                  (int((x + nw / 2) * w), int((y + nh / 2) * h)), colors[int(cls)], 2)

            # Draw Pred (Right)
            img_pred = img.copy()
            for b in self.model(str(img_path), verbose=False)[0].boxes:
                coords = b.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(img_pred, (coords[0], coords[1]), (coords[2], coords[3]), colors[int(b.cls)], 2)

            # Stitch
            canvas = np.zeros((h + 50, w * 2, 3), np.uint8)
            canvas[50:, :w], canvas[50:, w:] = img_gt, img_pred
            cv2.putText(canvas, "GT", (w // 2 - 20, 35), 0, 0.8, (255, 255, 255), 2)
            cv2.putText(canvas, "PRED", (w + w // 2 - 30, 35), 0, 0.8, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(save_path, img_path.name), canvas)
