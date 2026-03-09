from abc import ABC, abstractmethod
import os, json

import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.setup_experiment import get_exp_dir


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
        # Ultralytics engine handles the complex mAP logic for us
        abs_exp_dir = os.path.abspath(exp_dir)
        val_dir = os.path.join(os.path.basename(abs_exp_dir), 'val')
        metrics = self.model.val(
            data=data_cfg,
            imgsz=640,
            plots=True,
            project=os.path.dirname(abs_exp_dir),
            name=val_dir,
        )

        # Return structured results
        metrics_dict = {
            "map50": round(metrics.box.map50, 4),
            "map50_95": round(metrics.box.map, 4),
            "mean_precision": round(metrics.box.mp, 4),
            "mean_recall": round(metrics.box.mr, 4),
            "f1_score_benign": round(metrics.box.f1[0], 4),
            "f1_score_malignant": round(metrics.box.f1[1], 4),
            "f1_score": round((metrics.box.f1[0]+metrics.box.f1[1])/2, 4),
        }
        self._save_results(
            exp_dir=exp_dir,
            metrics=metrics_dict,
            confusion_matrix_data=metrics.confusion_matrix.matrix
        )
        return metrics_dict
