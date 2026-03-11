import os, argparse

from src.core.evaluator import YOLOEvaluator
from src.datasets.factory import get_dataloader
from src.models.registery import get_model
from src.utils.setup_experiment import get_exp_dir, get_best_weights


def validate():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name (e.g., yolo26n_ca)")
    parser.add_argument("-w", "--weights", type=str, default=None, help="Optional specific weights path")
    parser.add_argument("-d", "--data", type=str, default="data/")
    args = parser.parse_args()

    # 1. Automatically find weights and exp_dir
    weights_path = args.weights if args.weights else get_best_weights(args.model)
    exp_dir = get_exp_dir(args.model)

    print(f"🔍 Evaluating: {args.model}")
    print(f"📦 Using Weights: {weights_path}")

    # 2. Load model and data
    model = get_model(args.model, weights=weights_path)
    data_cfg = get_dataloader(args.model, args.data)

    # 3. Dispatch to Evaluator
    if 'yolo' in args.model.lower():
        evaluator = YOLOEvaluator(model, None)
        results = evaluator(data_cfg, exp_dir)
        print(f"🚀 Validation Complete for {args.model}")


if __name__ == "__main__":
    validate()