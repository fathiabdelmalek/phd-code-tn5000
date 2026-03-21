import argparse
import os
from pathlib import Path

from src.core.evaluators.registry import get_evaluator
from src.datasets.factory import get_dataloader
from src.models.registery import get_model
from src.utils.setup_experiment import get_exp_dir, get_best_weights


def validate():
    parser = argparse.ArgumentParser(description="Evaluate a detection model")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., yolo26n_ca, faster_rcnn, fcos)",
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default=None,
        help="Path to weights file (auto-discovers if not provided)",
    )
    parser.add_argument(
        "-d", "--data", type=str, default="data/", help="Path to data directory"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    args = parser.parse_args()

    weights_path = args.weights if args.weights else get_best_weights(args.model)
    if not weights_path or not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights not found. Please provide weights path or train the model first."
        )

    exp_dir = get_exp_dir(args.model)
    if exp_dir is None:
        exp_dir = f"experiments/{args.model}"

    print(f"Evaluating: {args.model}")
    print(f"Weights: {weights_path}")
    print(f"Results will be saved to: {exp_dir}")

    model = get_model(args.model, weights=weights_path)

    evaluator = get_evaluator(
        model_name=args.model,
        model=model,
        dataloader=None,
        data_cfg=get_dataloader(args.model, args.data, split="test")
        if "yolo" in args.model.lower()
        else None,
        exp_dir=exp_dir,
        conf_thres=args.conf,
        iou_thres=args.iou,
    )

    metrics = evaluator.evaluate()

    print(f"\nEvaluation Results:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    print(f"\nResults saved to: {exp_dir}/results/")


if __name__ == "__main__":
    validate()
