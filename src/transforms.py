"""
Data Augmentations using Albumentations.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


_SPATIAL = [
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, border_mode=0, p=0.5),
]

_PIXEL = [
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
]

_PIXEL_EXTENDED = [
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.GaussNoise(p=0.2),
    A.CLAHE(clip_limit=2.0, p=0.2),
]


def get_train_transforms(model: str = 'yolo') -> A.Compose:
    """
    Returns training augmentation pipeline.

    Args:
        model: 'frcnn' | 'fcos' | 'yolo'
    """
    if model == 'yolo':
        return A.Compose(
            _SPATIAL + _PIXEL,
            bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['labels'],
                min_visibility=0.3,
                min_area=100,
                clip=True,
            )
        )

    # Pascal VOC format (Faster R-CNN, SSD, etc.)
    return A.Compose(
        _SPATIAL + _PIXEL + [ToTensorV2()],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_visibility=0.3,
            min_area=100,
            clip=True,
        )
    )


def get_val_transforms() -> A.Compose:
    """
    Validation/test pipeline — no augmentation, just tensor conversion.
    Normalization should be applied separately if needed.
    """
    return A.Compose(
        [ToTensorV2()],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_visibility=0.0,   # keep all boxes at val time
            clip=True,
        )
    )
