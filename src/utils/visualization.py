import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
) -> None:
    """Plot and save a confusion matrix."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else ".0f",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_loss_curves(
    train_loss: List[float],
    val_loss: List[float],
    save_path: str,
    train_label: str = "Train Loss",
    val_label: str = "Val Loss",
    title: str = "Training Loss Curve",
) -> None:
    """Plot and save training/validation loss curves."""
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label=train_label, linewidth=2, marker="o")
    plt.plot(epochs, val_loss, label=val_label, linewidth=2, marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pr_curves(
    pr_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: str,
    title: str = "Precision-Recall Curves",
) -> None:
    """Plot and save Precision-Recall curves."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))

    for cls_name, (recall, precision) in pr_data.items():
        plt.plot(recall, precision, label=cls_name, linewidth=2)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics_comparison(
    metrics_dict: Dict[str, float],
    save_path: str,
    title: str = "Model Metrics",
) -> None:
    """Plot a bar chart comparing different metrics."""
    import matplotlib.pyplot as plt

    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, values, color="steelblue", edgecolor="black")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
        )

    plt.ylabel("Value")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics_by_epoch(
    history: Dict[str, List[float]],
    save_path: str,
    title: str = "Training History",
) -> None:
    """Plot metrics over epochs."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    for metric_name, values in history.items():
        epochs = range(1, len(values) + 1)
        plt.plot(epochs, values, label=metric_name, linewidth=2, marker="o")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_visualization(save_path: str, plot_type: str, data: Any, **kwargs) -> None:
    """Save a visualization based on type."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if plot_type == "confusion_matrix":
        plot_confusion_matrix(data, save_path=save_path, **kwargs)
    elif plot_type == "loss":
        plot_loss_curves(data["train"], data["val"], save_path=save_path, **kwargs)
    elif plot_type == "pr_curve":
        plot_pr_curves(data, save_path=save_path, **kwargs)
    elif plot_type == "metrics":
        plot_metrics_comparison(data, save_path=save_path, **kwargs)
    elif plot_type == "history":
        plot_metrics_by_epoch(data, save_path=save_path, **kwargs)
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")
