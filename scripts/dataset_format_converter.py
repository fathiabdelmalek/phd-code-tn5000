# ═══════════════════════════════════════════════════════════════════
# ONE-TIME YOLO CONVERSION — Run once, creates perfect structure
# ═══════════════════════════════════════════════════════════════════

import os
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ── Your PASCAL VOC datasets on Drive ────────────────────────────────
ROOT_DIR = '/home/fathi/dev/research/phd/code/tn5000/data'
VOC_ROOT = os.path.join(ROOT_DIR, 'voc')
VOC_IMAGES = os.path.join(VOC_ROOT, 'JPEGImages')
VOC_ANNOTS = os.path.join(VOC_ROOT, 'Annotations')
VOC_SPLITS = os.path.join(VOC_ROOT, 'ImageSets', 'Main')

# ── Where to create YOLO datasets ────────────────────────────────────
YOLO_ROOT = os.path.join(ROOT_DIR, 'yolo')
YAML_PATH = os.path.join(YOLO_ROOT, 'data.yaml')

# ── Create YOLO structure ───────────────────────────────────────────
print("Creating YOLO datasets structure...")

for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(YOLO_ROOT, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(YOLO_ROOT, 'labels', split), exist_ok=True)

print("Structure created:")
print(f"  {YOLO_ROOT}/")
print(f"    images/")
print(f"      train/  val/  test/")
print(f"    labels/")
print(f"      train/  val/  test/")


# ── Convert and copy each split ─────────────────────────────────────
def convert_voc_to_yolo_bbox(size, box):
    """Convert PASCAL VOC bbox to YOLO format (normalized cx, cy, w, h)"""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return (cx * dw, cy * dh, w * dw, h * dh)


for split in ['train', 'val', 'test']:
    print(f"\n{'=' * 60}")
    print(f"Processing {split} split...")
    print(f"{'=' * 60}")

    split_file = os.path.join(VOC_SPLITS, f'{split}.txt')
    with open(split_file) as f:
        image_ids = [line.strip() for line in f if line.strip()]

    print(f"Found {len(image_ids)} images in {split}.txt")

    converted = skipped = 0

    for img_id in tqdm(image_ids, desc=f"{split}"):
        img_src = os.path.join(VOC_IMAGES, f'{img_id}.jpg')
        xml_src = os.path.join(VOC_ANNOTS, f'{img_id}.xml')
        img_dest = os.path.join(YOLO_ROOT, 'images', split, f'{img_id}.jpg')  # ← updated
        lbl_dest = os.path.join(YOLO_ROOT, 'labels', split, f'{img_id}.txt')  # ← updated

        if not os.path.exists(img_src) or not os.path.exists(xml_src):
            skipped += 1
            continue

        tree = ET.parse(xml_src)
        root = tree.getroot()

        size_elem = root.find('size')
        width = int(size_elem.find('width').text)
        height = int(size_elem.find('height').text)

        yolo_lines = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            class_id = int(class_name)

            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            cx, cy, w, h = convert_voc_to_yolo_bbox(
                (width, height),
                (xmin, ymin, xmax, ymax)
            )

            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if not yolo_lines:
            skipped += 1
            continue

        with open(lbl_dest, 'w') as f:
            f.write('\n'.join(yolo_lines))

        shutil.copy2(img_src, img_dest)
        converted += 1

    print(f"\n{split}: {converted} samples converted, {skipped} skipped")

# ── Create data.yaml ────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("Creating data.yaml...")
print(f"{'=' * 60}")

yaml_content = f"""# TN5000 Thyroid Nodule Dataset (YOLO format)
# Converted from PASCAL VOC format

path: {YOLO_ROOT}  # datasets root dir
train: images/train  # ← updated
val: images/val      # ← updated
test: images/test    # ← updated

# Classes
nc: 2  # number of classes
names:
  0: benign
  1: malignant
"""

with open(YAML_PATH, 'w') as f:
    f.write(yaml_content)

print(f"data.yaml created at: {YAML_PATH}")
print("\nContents:")
print(yaml_content)

# ── Verification ────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("VERIFICATION")
print(f"{'=' * 60}")

for split in ['train', 'val', 'test']:
    img_dir = os.path.join(YOLO_ROOT, 'images', split)  # ← updated
    lbl_dir = os.path.join(YOLO_ROOT, 'labels', split)  # ← updated

    n_imgs = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    n_lbls = len([f for f in os.listdir(lbl_dir) if f.endswith('.txt')])

    match = '✅' if n_imgs == n_lbls else '❌'
    print(f"{split:5s}: {n_imgs} images, {n_lbls} labels {match}")

sample_lbl_dir = os.path.join(YOLO_ROOT, 'labels', 'train')  # ← updated
sample_lbl = os.listdir(sample_lbl_dir)[0]
sample_path = os.path.join(sample_lbl_dir, sample_lbl)

print(f"\nSample label ({sample_lbl}):")
with open(sample_path) as f:
    print(f.read())

print(f"\n{'=' * 60}")
print("✅ YOLO DATASET READY")
print(f"{'=' * 60}")
print(f"\nDataset location: {YOLO_ROOT}")
print(f"YAML file:        {YAML_PATH}")
print(f"\nUse this yaml in your training:")
print(f"  model.train(data='{YAML_PATH}', ...)")
