#!/usr/bin/env python3
"""
Simple Training Entry Point

Structure:
    experiments/{model}_{scale}_{timestamp}/
    ├── metadata.json
    ├── train/              # Training outputs
    │   ├── train_batch*.jpg
    │   └── labels.jpg
    └── val/               # Validation outputs (after evaluate.py)
        ├── boxes/
        ├── heatmaps/
        ├── plots/
        ├── results.json
        └── weights/

Usage:
    cd src
    python train.py yolo26 -s n -b 8 -e 50
    python train.py yolo26 -s m -b 4 -e 100
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
from datetime import datetime

from config import Config
from models import get_model, YOLO_SCALES, YOLO_SCALE_INFO
from trainers import YOLOTrainer, PyTorchTrainer


def get_experiment_dir(model_name, scale=None):
    """Create organized experiment directory."""

    if scale:
        exp_name = f"{model_name}_{scale}"
    else:
        exp_name = f"{model_name}"

    exp_dir = Path("experiments") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create train and val subdirectories
    train_dir = exp_dir / "val"
    val_dir = exp_dir / "val"

    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    # Val subdirs
    (val_dir / "boxes").mkdir(exist_ok=True)
    (val_dir / "heatmaps").mkdir(exist_ok=True)
    (val_dir / "plots").mkdir(exist_ok=True)
    (val_dir / "weights").mkdir(exist_ok=True)

    return exp_dir


def main():
    parser = argparse.ArgumentParser(description="Train a detection model")
    parser.add_argument(
        "model", type=str, choices=["yolo26", "fcos"], help="Model name"
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=str,
        default="n",
        choices=YOLO_SCALES,
        help=f"YOLO scale (default: n)",
    )
    parser.add_argument("--batch", "-b", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Number of epochs")
    parser.add_argument(
        "--preset",
        "-p",
        type=str,
        default="standard",
        choices=["fast", "standard", "adamw", "fine_tune"],
        help="Hyperparameter preset",
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer")
    parser.add_argument("--device", "-d", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    # Create experiment directory
    scale = args.scale if args.model == "yolo26" else None
    exp_dir = get_experiment_dir(args.model, scale)

    # Save metadata
    metadata = {
        "model": args.model,
        "scale": scale,
        "batch_size": args.batch,
        "epochs": args.epochs,
        "preset": args.preset,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    with open(exp_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Create config
    config = Config(
        batch_size=args.batch,
        epochs=args.epochs,
        preset=args.preset,
        device=args.device,
    )
    if args.lr:
        config.lr = args.lr
    if args.optimizer:
        config.optimizer = args.optimizer

    config.exp_dir = exp_dir

    # Load model
    if args.model == "yolo26":
        print(f"Loading YOLO26 ({YOLO_SCALE_INFO[args.scale]})")
        model = get_model("yolo26", num_classes=2, scale=args.scale)
    else:
        print(f"Loading {args.model.upper()}")
        model = get_model(args.model, num_classes=2, pretrained=True)

    # Get trainer
    trainer = (
        YOLOTrainer(model, config)
        if args.model == "yolo26"
        else PyTorchTrainer(model, config)
    )

    print(f"\nExperiment: {exp_dir}")
    print(f"Config: {config}")

    # Train
    results = trainer.train()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Experiment: {exp_dir}")
    print("\nRun evaluation with:")
    print(f"  python evaluate.py --exp {exp_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
