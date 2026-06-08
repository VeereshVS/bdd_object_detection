"""Main evaluation script for BDD100K Object Detection.

Usage (from project root bdd100k_object_detection/):
    python -m evaluation.run_evaluation --model model/runs/train/bdd100k_yolo11s/weights/best.pt --data-root /path/to/bdd100k

Or use a pretrained model name (auto-downloads):
    python evaluation/run_evaluation.py --model yolo11m --data-root /path/to/bdd100k

Available model variants: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root and sibling modules to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from evaluation.src.evaluate import (
    analyze_errors,
    evaluate_at_multiple_ious,
    evaluate_by_object_size,
    evaluate_detections,
)
from evaluation.src.visualize import (
    create_evaluation_report,
    plot_map_per_class,
    plot_performance_by_size,
    visualize_predictions_vs_gt,
)
from evaluation.src.analysis import (
    generate_full_analysis_report,
    generate_improvement_suggestions,
)
from model.src.model import BDD100KDetector
from model.src.dataset import BDD100K_CLASSES
from data_analysis.src.parser import BDD100KParser

logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main():
    """Run the evaluation pipeline."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Evaluate yolo11 on BDD100K validation set"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model weights or model name (e.g. yolo11s, yolo11m). Auto-downloads if not a local file.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory of BDD100K dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for evaluation",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of qualitative samples to visualize",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of validation images to evaluate (for quick testing)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "qualitative").mkdir(exist_ok=True)

    # Initialize model
    logger.info("Loading model from: %s", args.model)
    detector = BDD100KDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )

    # Load validation labels
    data_root = Path(args.data_root)
    val_labels_path = data_root / "labels" / "bdd100k_labels_images_val.json"

    logger.info("Loading validation labels: %s", val_labels_path)
    with open(val_labels_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    # Prepare ground truth and run predictions
    logger.info("Running inference on validation set...")
    predictions = []
    ground_truths = []

    val_parser = BDD100KParser(str(data_root))
    val_parser.parse_labels(str(val_labels_path), split="val")

    val_annotations = val_parser.annotations["val"]
    if args.max_images is not None:
        val_annotations = val_annotations[:args.max_images]
    logger.info("Evaluating on %d images", len(val_annotations))

    for ann in val_annotations:
        img_path = ann.image_path

        # Get ground truth
        gt_dets = []
        for det in ann.detections:
            gt_dets.append(
                {
                    "bbox": [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2],
                    "class_name": det.category,
                }
            )
        ground_truths.append({"image": ann.image_name, "detections": gt_dets})

        # Run prediction
        if Path(img_path).exists():
            result = detector.predict(img_path)
            predictions.append(
                {"image": ann.image_name, "detections": result["detections"]}
            )
        else:
            predictions.append({"image": ann.image_name, "detections": []})

        if len(predictions) % 500 == 0:
            logger.info("Processed %d images", len(predictions))

    # Evaluate
    logger.info("Computing evaluation metrics...")
    eval_results = evaluate_detections(
        predictions, ground_truths, iou_threshold=args.iou, classes=BDD100K_CLASSES
    )

    # Multi-IoU evaluation
    multi_iou_results = evaluate_at_multiple_ious(
        predictions, ground_truths, classes=BDD100K_CLASSES
    )

    # Error analysis
    logger.info("Analyzing errors...")
    error_analysis = analyze_errors(predictions, ground_truths, args.iou)

    # Size-based evaluation
    logger.info("Evaluating by object size...")
    size_results = evaluate_by_object_size(predictions, ground_truths, args.iou)

    # Generate visualizations
    logger.info("Generating visualizations...")
    plot_map_per_class(eval_results, str(output_dir / "plots" / "map_per_class.png"))
    plot_performance_by_size(
        size_results, str(output_dir / "plots" / "performance_by_size.png")
    )

    # Qualitative visualization
    logger.info("Creating qualitative visualizations...")
    for i, (pred, gt) in enumerate(zip(predictions[:args.num_samples], ground_truths)):
        img_path = str(data_root / "images" / "100k" / "val" / pred["image"])
        if Path(img_path).exists():
            visualize_predictions_vs_gt(
                img_path,
                pred["detections"],
                gt["detections"],
                str(output_dir / "qualitative" / f"sample_{i:04d}.jpg"),
            )

    # Generate reports
    logger.info("Generating reports...")
    full_report = generate_full_analysis_report(
        eval_results, error_analysis, size_results
    )

    with open(output_dir / "full_report.txt", "w", encoding="utf-8") as f:
        f.write(full_report)

    # Save JSON results
    results_json = {
        "eval_results": eval_results,
        "multi_iou": {
            "mAP_50": multi_iou_results["mAP_50"],
            "mAP_75": multi_iou_results["mAP_75"],
            "mAP_50_95": multi_iou_results["mAP_50_95"],
        },
        "error_summary": error_analysis["summary"],
        "size_performance": {k: v["mAP"] for k, v in size_results.items()},
        "suggestions": generate_improvement_suggestions(eval_results, error_analysis),
    }

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)

    print(full_report)
    logger.info("Evaluation complete! Results saved to: %s", output_dir)


if __name__ == "__main__":
    main()
