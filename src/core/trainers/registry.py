from ..base_trainer import BaseTrainer


def get_trainer(
    model_name: str,
    model=None,
    train_loader=None,
    val_loader=None,
    data_root: str = None,
    exp_dir: str = None,
    config: dict = None,
    weights: str = None,
    resume: bool = False,
    device: str = "cuda",
    callbacks: dict = None,
) -> BaseTrainer:
    """Factory function to get the appropriate trainer.

    Args:
        model_name: Name of the model (e.g., 'yolo26n_ca', 'faster_rcnn', 'fcos')
        model: Pre-initialized model (optional for YOLO)
        train_loader: Training data loader
        val_loader: Validation data loader
        data_root: Path to data directory
        exp_dir: Experiment directory path
        config: Configuration dictionary
        weights: Path to weights file
        resume: Whether to resume training
        device: Device to use for PyTorch models
        callbacks: Dictionary of callback functions

    Returns:
        Appropriate trainer instance
    """
    model_name_lower = model_name.lower()

    if "yolo" in model_name_lower:
        from .yolo_trainer import YOLOTrainer

        return YOLOTrainer(
            model_name=model_name,
            data_root=data_root,
            exp_dir=exp_dir,
            config=config,
            weights=weights,
            resume=resume,
            callbacks=callbacks,
        )

    elif "rcnn" in model_name_lower or "fcos" in model_name_lower:
        from .pytorch_trainer import PyTorchTrainer

        if model is None:
            from ...models.registery import get_model

            model = get_model(model_name, weights=weights)

        if train_loader is None or val_loader is None:
            from ...datasets.factory import get_dataloader

            batch_size = config.get("batch_size", 8) if config else 8
            train_loader = get_dataloader(
                model_name, data_root, split="train", batch_size=batch_size
            )
            val_loader = get_dataloader(
                model_name, data_root, split="val", batch_size=batch_size
            )

        return PyTorchTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            exp_dir=exp_dir,
            config=config,
            device=device,
            callbacks=callbacks,
        )

    else:
        raise ValueError(f"Unknown model type: {model_name}")
