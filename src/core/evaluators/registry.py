from ..base_evaluator import BaseEvaluator


def get_evaluator(
    model_name: str,
    model=None,
    dataloader=None,
    data_cfg: str = None,
    exp_dir: str = None,
    class_names: list = None,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    device: str = "cuda",
) -> BaseEvaluator:
    """Factory function to get the appropriate evaluator.

    Args:
        model_name: Name of the model (e.g., 'yolo26n_ca', 'faster_rcnn', 'fcos')
        model: The model instance
        dataloader: Data loader (for PyTorch models)
        data_cfg: Path to data config YAML (for YOLO models)
        exp_dir: Experiment directory path
        class_names: List of class names
        conf_thres: Confidence threshold
        iou_thres: IoU threshold
        device: Device for PyTorch models

    Returns:
        Appropriate evaluator instance
    """
    model_name_lower = model_name.lower()

    if "yolo" in model_name_lower:
        from .yolo_evaluator import YOLOEvaluator

        return YOLOEvaluator(
            model=model,
            data_cfg=data_cfg,
            exp_dir=exp_dir,
            class_names=class_names,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
        )

    elif "rcnn" in model_name_lower or "fcos" in model_name_lower:
        from .pytorch_evaluator import PyTorchEvaluator

        if dataloader is None:
            from ...datasets.factory import get_dataloader

            dataloader = get_dataloader(model_name, data_cfg or "data/", split="val")

        return PyTorchEvaluator(
            model=model,
            dataloader=dataloader,
            exp_dir=exp_dir,
            class_names=class_names,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            device=device,
        )

    else:
        raise ValueError(f"Unknown model type for evaluation: {model_name}")
