import os, argparse
from datetime import datetime

from .datasets.factory import get_dataloader
from .models.registery import get_model
from src.utils.transforms import get_train_transform
from src.utils.setup_experiment import setup_experiment


def train():
    # 1. Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="yolo26_ca, faster_rcnn, fcos, etc.")
    parser.add_argument("-e", "--epochs", type=int, default=50)
    parser.add_argument("-b", "--batch", type=int, default=8)
    parser.add_argument("-d", "--data", type=str, default="data/")
    parser.add_argument("-w", "--weights", type=str, default=None, help="Path to custom weights (.pt)")
    args = parser.parse_args()

    # 2. Setup automatic experiment folder
    exp_dir = setup_experiment(args.model)
    print(f"🚀 Training started. Results will be saved to: {exp_dir}")

    # 3. Get Data and Model
    data_cfg = get_dataloader(args.model, args.data)
    model = get_model(args.model, weights=args.weights)

    # 4. Dispatch training logic
    if 'yolo' in args.model.lower():
        abs_exp_dir = os.path.abspath(exp_dir)
        # Ultralytics engine
        model.train(
            data = data_cfg,
            batch = args.batch,
            epochs = args.epochs,
            optimizer = 'AdamW',
            weight_decay = 0.001,
            lr0 = 0.001,
            lrf = 0.01,
            cos_lr = True,
            dfl = 2.0,
            box = 10.0,
            cls = 1.5,
            patience = 20,
            # Augmentation
            augmentations = get_train_transform(),
            # Output
            project = os.path.dirname(abs_exp_dir),
            name = os.path.basename(abs_exp_dir),
            exist_ok = True,
            plots = True,
            save = True,
            resume = False,
        )
    else:
        # Custom PyTorch loop (Faster R-CNN/FCOS)
        # We will implement this loop next
        pass

if __name__ == "__main__":
    train()
