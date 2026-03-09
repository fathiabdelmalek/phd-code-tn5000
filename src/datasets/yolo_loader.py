import os
import yaml


class YOLODataLoader:
    def __init__(self, data_root):
        """
        Interfaces with the pre-converted YOLO datasets.
        data_root: Path to the 'data' folder (which contains 'yolo/')
        """
        self.yolo_root = os.path.join(data_root, 'yolo')
        self.yaml_path = os.path.join(self.yolo_root, 'data.yaml')

    def get_config_path(self):
        """
        Returns the absolute path to data.yaml.
        Crucial for YOLO because it often requires absolute paths to avoid confusion.
        """
        if not os.path.exists(self.yaml_path):
            raise FileNotFoundError(
                f"YOLO data.yaml not found at {self.yaml_path}. "
                "Please run scripts/dataset_format_converter.py first."
            )

        # Verify that the image directories actually contain data
        for split in ['train', 'val', 'test']:
            img_path = os.path.join(self.yolo_root, 'images', split)
            if not os.path.exists(img_path) or len(os.listdir(img_path)) == 0:
                print(f"Warning: YOLO {split} directory is empty or missing.")

        return os.path.abspath(self.yaml_path)

    def preview_stats(self):
        """Prints a quick summary of the converted YOLO datasets."""
        with open(self.yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        print(f"\n--- YOLO Dataset Config ---")
        print(f"Root: {data['path']}")
        print(f"Classes: {data['names']}")
        print(f"---------------------------\n")
