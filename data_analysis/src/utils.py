"""Utility functions for BDD100K data analysis."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the application.

    Args:
        level: Logging level.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def save_results_json(results: Dict[str, Any], output_path: str) -> None:
    """Save analysis results to a JSON file.

    Args:
        results: Dictionary with analysis results.
        output_path: Path to save the JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Results saved to %s", output_path)


def ensure_directory(path: str) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path.

    Returns:
        Path object for the directory.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_image_paths(
    data_root: str, split: str, extensions: List[str] = None
) -> List[str]:
    """Get all image paths for a given split.

    Args:
        data_root: Root directory of the dataset.
        split: Dataset split (train/val).
        extensions: List of valid image extensions.

    Returns:
        List of image file paths.
    """
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png"]

    image_dir = Path(data_root) / "images" / "100k" / split
    if not image_dir.exists():
        logger.warning("Image directory not found: %s", image_dir)
        return []

    image_paths = []
    for ext in extensions:
        image_paths.extend(str(p) for p in image_dir.glob(f"*{ext}"))

    return sorted(image_paths)


def validate_dataset_structure(data_root: str) -> Dict[str, bool]:
    """Validate the expected dataset directory structure.

    Args:
        data_root: Root directory of the dataset.

    Returns:
        Dictionary indicating which expected paths exist.
    """
    data_root = Path(data_root)
    expected_paths = {
        "labels/bdd100k_labels_images_train.json": (
            data_root / "labels" / "bdd100k_labels_images_train.json"
        ),
        "labels/bdd100k_labels_images_val.json": (
            data_root / "labels" / "bdd100k_labels_images_val.json"
        ),
        "images/100k/train": data_root / "images" / "100k" / "train",
        "images/100k/val": data_root / "images" / "100k" / "val",
    }

    results = {}
    for name, path in expected_paths.items():
        results[name] = path.exists()
        if not path.exists():
            logger.warning("Missing expected path: %s", path)

    return results
