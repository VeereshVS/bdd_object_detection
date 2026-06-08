"""Training pipeline for yolo11 on BDD100K dataset.

This module provides the complete training pipeline including:
- Dataset preparation (BDD100K to YOLO format conversion)
- Model initialization
- Training loop execution
- Checkpoint saving and logging

Usage:
    python train.py --bdd-root /path/to/bdd100k --epochs 1 --subset 1000
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from .dataset import prepare_yolo_dataset, BDD100K_CLASSES
from .model import BDD100KDetector

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def train_pipeline(
    bdd_root: str,
    output_dir: str = "runs",
    model_variant: str = "yolo11s",
    epochs: int = 1,
    batch_size: int = 16,
    img_size: int = 640,
    subset_size: int = None,
    device: str = None,
) -> str:
    """Execute the complete training pipeline.

    Steps:
    1. Convert BDD100K annotations to YOLO format
    2. Initialize yolo11 model with pretrained weights
    3. Train for specified epochs
    4. Save results and best weights

    Args:
        bdd_root: Root directory of BDD100K dataset.
        output_dir: Directory for training outputs.
        model_variant: yolo11 variant to use.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        img_size: Input image size.
        subset_size: Number of images to use (for quick testing).
        device: Device (cuda/cpu/mps). Auto-detected if None.

    Returns:
        Path to the best model weights.
    """
    setup_logging()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Prepare dataset
    logger.info("=" * 60)
    logger.info("STEP 1: Preparing dataset (BDD100K -> YOLO format)")
    logger.info("=" * 60)

    dataset_dir = str(output_path / "dataset")
    yaml_path = prepare_yolo_dataset(
        bdd100k_root=bdd_root,
        output_dir=dataset_dir,
        subset_size=subset_size,
    )
    logger.info("Dataset YAML: %s", yaml_path)

    # Step 2: Initialize model
    logger.info("=" * 60)
    logger.info("STEP 2: Initializing %s model", model_variant)
    logger.info("=" * 60)

    detector = BDD100KDetector(
        model_variant=model_variant,
        conf_threshold=0.25,
        iou_threshold=0.45,
    )

    # Auto-detect device
    if device is None:
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except ImportError:
            device = "cpu"

    logger.info("Using device: %s", device)
    logger.info("Model variant: %s", model_variant)
    logger.info("Epochs: %d", epochs)
    logger.info("Batch size: %d", batch_size)
    logger.info("Image size: %d", img_size)

    # Step 3: Train
    logger.info("=" * 60)
    logger.info("STEP 3: Training")
    logger.info("=" * 60)

    start_time = time.time()

    best_weights = detector.train(
        dataset_yaml=yaml_path,
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
        project=str(output_path / "train"),
        name=f"bdd100k_{model_variant}",
        device=device,
    )

    elapsed = time.time() - start_time
    logger.info("Training complete in %.1f seconds", elapsed)
    logger.info("Best weights saved to: %s", best_weights)

    # Step 4: Quick validation
    logger.info("=" * 60)
    logger.info("STEP 4: Validation")
    logger.info("=" * 60)

    metrics = detector.validate(
        dataset_yaml=yaml_path,
        batch_size=batch_size,
        img_size=img_size,
        device=device,
    )

    logger.info("Validation Results:")
    logger.info("  mAP@0.5: %.4f", metrics["mAP50"])
    logger.info("  mAP@0.5:0.95: %.4f", metrics["mAP50-95"])
    logger.info("  Precision: %.4f", metrics["precision"])
    logger.info("  Recall: %.4f", metrics["recall"])

    if metrics["per_class_ap50"]:
        logger.info("  Per-class AP@0.5:")
        for cls, ap in metrics["per_class_ap50"].items():
            logger.info("    %s: %.4f", cls, ap)

    return best_weights


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Train yolo11 on BDD100K dataset"
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
        default="runs",
        help="Output directory for training results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11s",
        choices=["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"],
        help="yolo11 model variant",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image size",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Subset size for quick training (number of images)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu/mps). Auto-detected if not specified.",
    )

    args = parser.parse_args()

    train_pipeline(
        bdd_root=args.bdd_root,
        output_dir=args.output,
        model_variant=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        subset_size=args.subset,
        device=args.device,
    )


if __name__ == "__main__":
    main()
