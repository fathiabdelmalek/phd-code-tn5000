import os
import torch
import cv2
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset


class ThyroidVOCDataset(Dataset):
    """
    Pascal VOC-format datasets for thyroid nodule detection.
    Labels in XML: 0 = benign, 1 = malignant
    Labels returned: 1 = benign, 2 = malignant (0 reserved for background)
    """
    def __init__(self, root_dir, split='train', shift=False, transforms=None):
        self.img_dir    = os.path.join(root_dir, 'voc/JPEGImages')
        self.ann_dir    = os.path.join(root_dir, 'voc/Annotations')
        self.transforms = transforms
        self.shift      = shift

        split_file = os.path.join(root_dir, f'voc/ImageSets/Main/{split}.txt')
        with open(split_file, 'r') as f:
            self.ids = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id   = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        ann_path = os.path.join(self.ann_dir, f"{img_id}.xml")

        # Load image with cv2 (BGR → RGB)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        annotation = self._parse_xml(ann_path)
        boxes  = annotation['boxes']   # list of [xmin, ymin, xmax, ymax]
        labels = annotation['labels']  # list of ints (already +1 shifted)

        if self.transforms:
            # Albumentations-style: expects image as np.ndarray
            out    = self.transforms(image=image, bboxes=boxes, labels=labels)
            image  = out['image']                      # tensor after ToTensorV2
            if image.dtype == torch.uint8:
                image = image.float() / 255.0
            boxes  = out['bboxes']
            labels = out['labels']

        # Convert to tensors (handles empty annotation edge case)
        if len(boxes) == 0:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.int64)
        else:
            boxes_t  = torch.as_tensor(boxes,  dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)

        if not self.transforms:
            # No albumentations: convert image manually
            image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        area = (
            (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
            if len(boxes_t) > 0
            else torch.zeros((0,), dtype=torch.float32)
        )

        target = {
            'boxes':    boxes_t,
            'labels':   labels_t,
            'image_id': torch.tensor([idx]),
            'area':     area,
            'iscrowd':  torch.zeros((len(labels_t),), dtype=torch.int64),
        }

        return image, target

    def _parse_xml(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Annotation not found: {path}")

        root   = ET.parse(path).getroot()
        boxes, labels = [], []

        for obj in root.iter('object'):
            # Shift label: 0 (benign) → 1, 1 (malignant) → 2
            # 0 is reserved for background in Faster R-CNN etc.
            label = int(obj.find('name').text) + 1 if self.shift else int(obj.find('name').text)

            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # Skip degenerate boxes
            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        return {'boxes': boxes, 'labels': labels}
