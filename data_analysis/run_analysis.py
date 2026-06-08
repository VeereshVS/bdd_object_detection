"""Main entry point for BDD100K data analysis.

This script performs comprehensive analysis of the BDD100K dataset
for the object detection task, generating visualizations and reports.

Usage:
    python run_analysis.py --data-root /path/to/bdd100k --output /path/to/output
"""

import argparse
import logging
import sys
from pathlib import Path

from src.analyzer import BDD100KAnalyzer
from src.parser import BDD100KParser
from src.utils import ensure_directory, save_results_json, setup_logging
from src.visualizer import (
    plot_attribute_distribution,
    plot_bbox_statistics,
    plot_class_distribution,
    plot_cooccurrence_matrix,
    plot_objects_per_image_histogram,
    plot_occlusion_truncation_stats,
    plot_spatial_heatmap,
    plot_train_val_comparison,
    visualize_sample,
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="BDD100K Object Detection Data Analysis"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/data/bdd100k",
        help="Root directory of the BDD100K dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/output",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--train-labels",
        type=str,
        default=None,
        help="Path to training labels JSON (overrides default)",
    )
    parser.add_argument(
        "--val-labels",
        type=str,
        default=None,
        help="Path to validation labels JSON (overrides default)",
    )
    parser.add_argument(
        "--visualize-samples",
        type=int,
        default=10,
        help="Number of sample images to visualize",
    )
    return parser.parse_args()


def main():
    """Run the complete data analysis pipeline."""
    setup_logging()
    args = parse_args()

    data_root = Path(args.data_root)
    output_dir = ensure_directory(args.output)
    plots_dir = ensure_directory(str(output_dir / "plots"))
    samples_dir = ensure_directory(str(output_dir / "samples"))

    logger.info("Starting BDD100K data analysis")
    logger.info("Data root: %s", data_root)
    logger.info("Output directory: %s", output_dir)

    # Initialize parser
    parser = BDD100KParser(str(data_root))

    # Determine label paths
    train_labels = args.train_labels or str(
        data_root / "labels" / "bdd100k_labels_images_train.json"
    )
    val_labels = args.val_labels or str(
        data_root / "labels" / "bdd100k_labels_images_val.json"
    )

    # Parse labels
    logger.info("Parsing training labels: %s", train_labels)
    parser.parse_labels(train_labels, split="train")
    logger.info(
        "Parsed %d training images with detections",
        len(parser.annotations["train"]),
    )

    logger.info("Parsing validation labels: %s", val_labels)
    parser.parse_labels(val_labels, split="val")
    logger.info(
        "Parsed %d validation images with detections",
        len(parser.annotations["val"]),
    )

    # Initialize analyzer
    analyzer = BDD100KAnalyzer(parser)

    # 1. Class Distribution Analysis
    logger.info("Analyzing class distributions...")
    train_dist = analyzer.analyze_class_distribution("train")
    val_dist = analyzer.analyze_class_distribution("val")

    print("\n" + "=" * 60)
    print("TRAINING SET CLASS DISTRIBUTION")
    print("=" * 60)
    print(train_dist.to_string(index=False))

    print("\n" + "=" * 60)
    print("VALIDATION SET CLASS DISTRIBUTION")
    print("=" * 60)
    print(val_dist.to_string(index=False))

    plot_class_distribution(
        train_dist,
        title="Training Set",
        save_path=str(plots_dir / "train_class_distribution.png"),
    )
    plot_class_distribution(
        val_dist,
        title="Validation Set",
        save_path=str(plots_dir / "val_class_distribution.png"),
    )

    # 2. Train vs Val Comparison
    logger.info("Comparing train and validation splits...")
    comparison = analyzer.analyze_train_val_split()

    print("\n" + "=" * 60)
    print("TRAIN VS VALIDATION COMPARISON")
    print("=" * 60)
    print(comparison.to_string(index=False))

    plot_train_val_comparison(
        comparison, save_path=str(plots_dir / "train_val_comparison.png")
    )

    # 3. Bounding Box Statistics
    logger.info("Analyzing bounding box statistics...")
    train_bbox_stats = parser.get_bbox_stats("train")

    plot_bbox_statistics(
        train_bbox_stats, save_path=str(plots_dir / "bbox_statistics.png")
    )

    # 4. Anomaly Detection
    logger.info("Detecting anomalies...")
    train_anomalies = analyzer.detect_anomalies("train")
    val_anomalies = analyzer.detect_anomalies("val")

    print("\n" + "=" * 60)
    print("ANOMALY DETECTION SUMMARY (Training)")
    print("=" * 60)
    for key, value in train_anomalies["summary"].items():
        print(f"  {key}: {value}")

    # 5. Class Co-occurrence
    logger.info("Analyzing class co-occurrence...")
    cooccurrence = analyzer.analyze_class_cooccurrence("train")
    plot_cooccurrence_matrix(
        cooccurrence, save_path=str(plots_dir / "cooccurrence_matrix.png")
    )

    # 6. Attribute Distribution
    logger.info("Analyzing attribute distributions...")
    attr_dist = parser.get_attribute_distribution("train")
    plot_attribute_distribution(
        attr_dist, save_path=str(plots_dir / "attribute_distribution.png")
    )

    # 7. Objects per Image Histogram
    logger.info("Plotting objects per image distribution...")
    plot_objects_per_image_histogram(
        parser.annotations["train"],
        save_path=str(plots_dir / "objects_per_image.png"),
    )

    # 8. Occlusion/Truncation Stats
    logger.info("Analyzing occlusion and truncation...")
    occ_stats = parser.get_occlusion_truncation_stats("train")
    if occ_stats:
        plot_occlusion_truncation_stats(
            occ_stats, save_path=str(plots_dir / "occlusion_truncation.png")
        )

    # 9. Spatial Heatmaps
    logger.info("Generating spatial heatmaps...")
    for cls in ["car", "pedestrian", "traffic light", "traffic sign"]:
        plot_spatial_heatmap(
            parser.annotations["train"],
            cls,
            save_path=str(plots_dir / f"spatial_heatmap_{cls.replace(' ', '_')}.png"),
        )

    # 10. Interesting Samples
    logger.info("Finding interesting samples...")
    interesting = analyzer.find_interesting_samples("train", top_k=10)

    # Visualize some interesting samples
    logger.info("Visualizing sample images...")
    for category, samples in interesting.items():
        for i, sample in enumerate(samples[:args.visualize_samples]):
            image_name = sample.get("image", "")
            img_path = str(data_root / "images" / "100k" / "train" / image_name)

            # Find the corresponding annotation
            for ann in parser.annotations["train"]:
                if ann.image_name == image_name:
                    save_path = str(
                        samples_dir / f"{category}_{i}_{image_name}"
                    )
                    visualize_sample(img_path, ann, save_path=save_path)
                    break

    # 11. Analysis by Attributes
    logger.info("Analyzing detections by image attributes...")
    attr_analysis = analyzer.analyze_by_attributes("train")
    for attr_name, df in attr_analysis.items():
        df.to_csv(str(output_dir / f"detections_by_{attr_name}.csv"))

    # 12. Generate Summary Report
    logger.info("Generating summary report...")
    report = analyzer.generate_summary_report()
    save_results_json(report, str(output_dir / "analysis_report.json"))

    # Save anomalies
    save_results_json(
        {
            "train": train_anomalies["summary"],
            "val": val_anomalies["summary"],
        },
        str(output_dir / "anomalies_summary.json"),
    )

    # Save interesting samples info
    save_results_json(
        {k: v[:10] for k, v in interesting.items()},
        str(output_dir / "interesting_samples.json"),
    )

    logger.info("Analysis complete! Results saved to: %s", output_dir)
    print(f"\n{'=' * 60}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 60}")
    print(f"Results saved to: {output_dir}")
    print(f"Plots saved to: {plots_dir}")
    print(f"Sample visualizations saved to: {samples_dir}")


if __name__ == "__main__":
    main()
