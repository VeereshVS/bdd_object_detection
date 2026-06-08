"""BDD100K Dataset Loader for yolo11.

This module provides dataset loading utilities to convert BDD100K annotations
into the format required by yolo11 (Ultralytics) for training and inference.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# BDD100K detection classes mapped to YOLO class indices
BDD100K_CLASSES = [
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
]

CLASS_TO_ID = {cls: idx for idx, cls in enumerate(BDD100K_CLASSES)}

# Alternative names mapping
CLASS_NAME_MAP = {
    "person": "pedestrian",
    "light": "traffic light",
    "sign": "traffic sign",
}


def convert_bbox_to_yolo(
    bbox: Dict[str, float], img_width: int = 1280, img_height: int = 720
) -> Tuple[float, float, float, float]:
    """Convert BDD100K bbox format to YOLO format.

    BDD100K format: x1, y1, x2, y2 (absolute coordinates)
    YOLO format: x_center, y_center, width, height (normalized 0-1)

    Args:
        bbox: Dictionary with x1, y1, x2, y2 keys.
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        Tuple of (x_center, y_center, width, height) normalized to [0, 1].
    """
    x1 = bbox["x1"]
    y1 = bbox["y1"]
    x2 = bbox["x2"]
    y2 = bbox["y2"]

    # Calculate center and dimensions
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height

    # Clamp values to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))

    return x_center, y_center, width, height


def convert_bdd100k_to_yolo(
    labels_json_path: str,
    output_labels_dir: str,
    split: str = "train",
    subset_size: Optional[int] = None,
) -> List[str]:
    """Convert BDD100K annotations to YOLO format label files.

    Creates one .txt file per image with YOLO format annotations:
    class_id x_center y_center width height

    Args:
        labels_json_path: Path to BDD100K JSON labels file.
        output_labels_dir: Output directory for YOLO label files.
        split: Dataset split (train/val).
        subset_size: If specified, only convert this many images.

    Returns:
        List of image filenames that were converted.
    """
    output_dir = Path(output_labels_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(labels_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if subset_size:
        data = data[:subset_size]

    converted_images = []

    for item in data:
        image_name = item.get("name", "")
        labels = item.get("labels", [])

        if not labels:
            continue

        # Generate YOLO format annotations
        yolo_lines = []
        for label in labels:
            category = label.get("category", "")
            category = CLASS_NAME_MAP.get(category, category)

            if category not in CLASS_TO_ID:
                continue

            box2d = label.get("box2d", None)
            if box2d is None:
                continue

            class_id = CLASS_TO_ID[category]
            x_center, y_center, width, height = convert_bbox_to_yolo(box2d)

            # Skip invalid boxes
            if width <= 0 or height <= 0:
                continue

            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        if yolo_lines:
            # Write label file (same name as image but .txt extension)
            label_filename = Path(image_name).stem + ".txt"
            label_path = output_dir / label_filename

            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))

            converted_images.append(image_name)

    return converted_images


def create_yolo_dataset_yaml(
    output_path: str,
    train_images_dir: str,
    val_images_dir: str,
    train_labels_dir: str,
    val_labels_dir: str,
    dataset_root: Optional[str] = None,
) -> str:
    """Create YOLO dataset configuration YAML file.

    Args:
        output_path: Path to save the YAML file.
        train_images_dir: Path to training images.
        val_images_dir: Path to validation images.
        train_labels_dir: Path to training labels.
        val_labels_dir: Path to validation labels.
        dataset_root: Absolute path to dataset root directory.

    Returns:
        Path to the created YAML file.
    """
    if dataset_root is None:
        dataset_root = str(Path(output_path).parent.resolve())

    yaml_content = f"""# BDD100K Object Detection Dataset Configuration
# Auto-generated for yolo11 training

path: {dataset_root}
train: {train_images_dir}
val: {val_images_dir}

# Training labels
train_labels: {train_labels_dir}
val_labels: {val_labels_dir}

# Number of classes
nc: {len(BDD100K_CLASSES)}

# Class names
names:
"""
    for idx, cls_name in enumerate(BDD100K_CLASSES):
        yaml_content += f"  {idx}: {cls_name}\n"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    return str(output_path)


def prepare_yolo_dataset(
    bdd100k_root: str,
    output_dir: str,
    subset_size: Optional[int] = None,
) -> str:
    """Prepare the complete YOLO dataset structure from BDD100K.

    Creates the following structure:
    output_dir/
    ├── dataset.yaml
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/

    Args:
        bdd100k_root: Root directory of BDD100K dataset.
        output_dir: Output directory for YOLO format dataset.
        subset_size: If specified, only use this many images per split.

    Returns:
        Path to the dataset YAML configuration file.
    """
    bdd_root = Path(bdd100k_root)
    out_root = Path(output_dir)

    # Create directories
    (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # Convert training labels
    train_json = str(bdd_root / "labels" / "bdd100k_labels_images_train.json")
    train_images = convert_bdd100k_to_yolo(
        train_json,
        str(out_root / "labels" / "train"),
        split="train",
        subset_size=subset_size,
    )
    print(f"Converted {len(train_images)} training label files")

    # Convert validation labels
    val_json = str(bdd_root / "labels" / "bdd100k_labels_images_val.json")
    val_images = convert_bdd100k_to_yolo(
        val_json,
        str(out_root / "labels" / "val"),
        split="val",
        subset_size=subset_size,
    )
    print(f"Converted {len(val_images)} validation label files")

    # Create symlinks or copy images
    bdd_train_imgs = bdd_root / "images" / "100k" / "train"
    bdd_val_imgs = bdd_root / "images" / "100k" / "val"

    print("Linking/copying training images...")
    for img_name in train_images:
        src = bdd_train_imgs / img_name
        dst = out_root / "images" / "train" / img_name
        if src.exists() and not dst.exists():
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(str(src), str(dst))

    print("Linking/copying validation images...")
    for img_name in val_images:
        src = bdd_val_imgs / img_name
        dst = out_root / "images" / "val" / img_name
        if src.exists() and not dst.exists():
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(str(src), str(dst))

    # Create dataset YAML
    yaml_path = create_yolo_dataset_yaml(
        str(out_root / "dataset.yaml"),
        train_images_dir="images/train",
        val_images_dir="images/val",
        train_labels_dir="labels/train",
        val_labels_dir="labels/val",
        dataset_root=str(out_root.resolve()),
    )

    print(f"Dataset prepared at: {out_root}")
    print(f"YAML config: {yaml_path}")

    return yaml_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert BDD100K to YOLO format"
    )
    parser.add_argument(
        "--bdd-root",
        type=str,
        required=True,
        help="Root directory of BDD100K dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for YOLO format dataset",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Number of images to use (for quick testing)",
    )

    args = parser.parse_args()
    prepare_yolo_dataset(args.bdd_root, args.output, args.subset_size)
