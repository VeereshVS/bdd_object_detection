"""Inference utilities for BDD100K object detection.

Provides single image and batch inference with visualization support.

Usage:
    python inference.py --model weights/best.pt --source /path/to/images
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .model import BDD100KDetector

logger = logging.getLogger(__name__)

# Visualization colors (BGR for OpenCV)
COLORS = {
    "pedestrian": (0, 0, 255),
    "rider": (0, 128, 255),
    "car": (0, 255, 0),
    "truck": (255, 0, 0),
    "bus": (0, 255, 255),
    "train": (255, 0, 255),
    "motorcycle": (255, 255, 0),
    "bicycle": (255, 0, 128),
    "traffic light": (0, 255, 128),
    "traffic sign": (128, 255, 0),
}


def draw_detections(
    image: np.ndarray,
    detections: List[Dict],
    show_conf: bool = True,
) -> np.ndarray:
    """Draw detection results on an image.

    Args:
        image: Input image (BGR format).
        detections: List of detection dictionaries.
        show_conf: Whether to show confidence scores.

    Returns:
        Image with drawn detections.
    """
    img = image.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
        cls_name = det["class_name"]
        conf = det["confidence"]
        color = COLORS.get(cls_name, (255, 255, 255))

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{cls_name}"
        if show_conf:
            label += f" {conf:.2f}"

        font_scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        cv2.rectangle(
            img,
            (x1, y1 - text_h - baseline - 4),
            (x1 + text_w, y1),
            color,
            -1,
        )
        cv2.putText(
            img,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    return img


def run_inference(
    model_path: str,
    source: str,
    output_dir: str = "inference_results",
    conf_threshold: float = 0.25,
    save_images: bool = True,
    save_json: bool = True,
    max_images: Optional[int] = None,
) -> List[Dict]:
    """Run inference on images and save results.

    Args:
        model_path: Path to model weights.
        source: Path to image or directory of images.
        output_dir: Output directory for results.
        conf_threshold: Confidence threshold.
        save_images: Whether to save annotated images.
        save_json: Whether to save results as JSON.
        max_images: Maximum number of images to process.

    Returns:
        List of detection results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize detector
    detector = BDD100KDetector(
        model_path=model_path, conf_threshold=conf_threshold
    )

    # Get image paths
    source_path = Path(source)
    if source_path.is_file():
        image_paths = [str(source_path)]
    else:
        extensions = ["*.jpg", "*.jpeg", "*.png"]
        image_paths = []
        for ext in extensions:
            image_paths.extend(str(p) for p in source_path.glob(ext))
        image_paths.sort()

    if max_images is not None:
        image_paths = image_paths[:max_images]

    logger.info("Running inference on %d images", len(image_paths))

    all_results = []
    for img_path in image_paths:
        result = detector.predict(img_path)
        all_results.append(result)

        if save_images and result["detections"]:
            img = cv2.imread(img_path)
            if img is not None:
                annotated = draw_detections(img, result["detections"])
                save_path = str(
                    output_path / f"det_{Path(img_path).name}"
                )
                cv2.imwrite(save_path, annotated)

    if save_json:
        json_path = str(output_path / "results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Results saved to: %s", json_path)

    return all_results


def main():
    """Main entry point for inference script."""
    parser = argparse.ArgumentParser(
        description="Run yolo11 inference on BDD100K images"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model weights",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to image or directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inference_results",
        help="Output directory",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to run inference on",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_inference(
        model_path=args.model,
        source=args.source,
        output_dir=args.output,
        conf_threshold=args.conf,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
