import os, argparse

from src.core.evaluator import YOLOEvaluator
from src.datasets.factory import get_dataloader
from src.models.registery import get_model
from src.utils.setup_experiment import get_exp_dir


def validate():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-w", "--weights", type=str, default=None)
    parser.add_argument("-d", "--data", type=str, default="data/")
    args = parser.parse_args()

    # Load model and data
    model = get_model(args.model, weights=args.weights if args.weights else get_exp_dir(args.model))
    data_cfg = get_dataloader(args.model, args.data, split='val')
    exp_dir = os.path.join("experiments", args.model)

    # Dispatch to specific Evaluator
    if 'yolo' in args.model.lower():
        evaluator = YOLOEvaluator(model, None)
        results = evaluator(data_cfg, exp_dir)
        print(f"🚀 Validation Complete: {results}")

if __name__ == "__main__":
    validate()
