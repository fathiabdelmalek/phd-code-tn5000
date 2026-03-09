from torch.utils.data import DataLoader

from .voc_loader import ThyroidVOCDataset
from .yolo_loader import YOLODataLoader
from src.utils.transforms import get_train_transform, get_val_transform


def get_dataloader(model_type, data_root, split='train', shift=False, batch_size=8):
    """
    Unified Data Entry Point.
    - If YOLO: returns the .yaml file path.
    - If Others: returns a PyTorch DataLoader.
    """
    if 'yolo' in model_type.lower():
        # YOLO (Ultralytics) engine handles its own internal augmentations
        # via the yaml configuration. We do not pass Albumentations here.
        return YOLODataLoader(data_root).get_config_path()

    # Standard PyTorch workflow (FCOS, Faster R-CNN, etc.)
    transform = get_train_transform('frcnn') if split == 'train' else get_val_transform()
    dataset = ThyroidVOCDataset(root_dir=data_root, split=split, shift=shift, transforms=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda x: tuple(zip(*x))
    )
