import argparse

from src.core.trainers.registry import get_trainer
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.setup_experiment import setup_experiment


def train():
    parser = argparse.ArgumentParser(description="Train a detection model")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., yolo26n_ca, faster_rcnn, fcos)",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=None, help="Number of training epochs"
    )
    parser.add_argument("-b", "--batch", type=int, default=4, help="Batch size")
    parser.add_argument(
        "-d", "--data", type=str, default="data/", help="Path to data directory"
    )
    parser.add_argument(
        "-w", "--weights", type=str, default=None, help="Path to custom weights file"
    )
    parser.add_argument(
        "-r", "--resume", action="store_true", help="Resume from last checkpoint"
    )
    parser.add_argument(
        "-c", "--config", type=str, default=None, help="Path to config YAML file"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.epochs is not None:
        config._data["epochs"] = args.epochs
    if args.batch is not None:
        config._data["batch_size"] = args.batch

    exp_dir = setup_experiment(args.model)
    logger = get_logger(exp_dir, name="train")

    logger.info(f"Starting training: {args.model}")
    logger.info(f"Experiment directory: {exp_dir}")

    trainer = get_trainer(
        model_name=args.model,
        data_root=args.data,
        exp_dir=exp_dir,
        config=config.to_dict(),
        weights=args.weights,
        resume=args.resume,
    )

    results = trainer.train()

    logger.info(f"Training complete!")
    logger.info(f"Results: {results}")

    print(f"\nTraining complete. Results saved to: {exp_dir}")


if __name__ == "__main__":
    train()
