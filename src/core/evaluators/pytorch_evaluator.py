from typing import Dict, List, Optional, Any
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from ..base_evaluator import BaseEvaluator


class PyTorchEvaluator(BaseEvaluator):
    """Evaluator implementation for PyTorch models (Faster-RCNN, FCOS, etc.)."""

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        exp_dir: str,
        class_names: Optional[List[str]] = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        device: str = "cuda",
    ):
        super().__init__(
            model=model,
            dataloader=dataloader,
            exp_dir=exp_dir,
            class_names=class_names or ["benign", "malignant"],
            conf_thres=conf_thres,
            iou_thres=iou_thres,
        )
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.num_classes = len(self.class_names) + 1

    def evaluate(self) -> Dict[str, float]:
        """Run full evaluation on the test set."""
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, targets in tqdm(self.dataloader, desc="Evaluating"):
                images = [img.to(self.device) for img in images]
                targets = [
                    {k: v.to(self.device) for k, v in t.items()} for t in targets
                ]

                outputs = self.model(images)

                all_predictions.extend(outputs)
                all_targets.extend(targets)

        metrics = self._compute_metrics(all_predictions, all_targets)
        self._last_metrics = metrics

        confusion_matrix = self._compute_confusion_matrix(all_predictions, all_targets)

        self.save_results(metrics, extra_data={"confusion_matrix": confusion_matrix})

        return metrics

    def predict(self, *args, **kwargs):
        """Run inference on a single image or batch."""
        self.model.eval()
        with torch.no_grad():
            return self.model(*args, **kwargs)

    def get_metrics(self) -> Dict[str, float]:
        """Return the most recent evaluation metrics."""
        return self._last_metrics

    def _compute_metrics(self, predictions: List, targets: List) -> Dict[str, float]:
        """Compute evaluation metrics."""
        tp, fp, fn = 0, 0, 0

        for pred, target in zip(predictions, targets):
            pred_labels = pred["labels"].cpu().numpy()
            target_labels = target["labels"].cpu().numpy()

            pred_set = set(pred_labels)
            target_set = set(target_labels)

            tp += len(pred_set & target_set)
            fp += len(pred_set - target_set)
            fn += len(target_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        }

    def _compute_confusion_matrix(
        self, predictions: List, targets: List
    ) -> List[List[int]]:
        """Compute confusion matrix."""
        n = len(self.class_names) + 1
        cm = np.zeros((n, n), dtype=int)

        for pred, target in zip(predictions, targets):
            pred_labels = pred["labels"].cpu().numpy()
            target_labels = target["labels"].cpu().numpy()

            for t in target_labels:
                if len(pred_labels) > 0:
                    p = pred_labels[0]
                    cm[t, p] += 1
                else:
                    cm[t, 0] += 1

        return cm.tolist()
