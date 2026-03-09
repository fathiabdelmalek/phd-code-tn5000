import json
import os
import matplotlib.pyplot as plt
import seaborn as sns


def save_results(exp_dir, metrics, confusion_matrix_data, train_loss=None, val_loss=None):
    """
    Standardized export for all models.
    """
    results_dir = os.path.join(exp_dir, "results")

    # 1. Save metrics.json
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # 2. Save Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_data, annot=True, fmt='d', cmap='Blues')
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
