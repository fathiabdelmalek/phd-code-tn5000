from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    """Abstract base class for all dataset implementations.

    All datasets must inherit from this class to ensure consistent interface
    across the pipeline.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        class_names: Optional[List[str]] = None,
    ):
        self.root_dir = root_dir
        self.split = split
        self.class_names = class_names or ["benign", "malignant"]

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """Get a single sample.

        Returns:
            Tuple of (image, target) where target is a dictionary containing:
            - boxes: Tensor of shape (N, 4) in xyxy format
            - labels: Tensor of shape (N,)
            - image_id: Tensor of shape (1,)
            - area: Tensor of shape (N,)
            - iscrowd: Tensor of shape (N,)
        """
        pass

    def get_class_names(self) -> List[str]:
        """Return the list of class names."""
        return self.class_names

    def get_num_classes(self) -> int:
        """Return the number of classes."""
        return len(self.class_names)

    def get_split(self) -> str:
        """Return the current split (train/val/test)."""
        return self.split
