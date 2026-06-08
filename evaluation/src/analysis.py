"""Performance analysis and failure clustering for BDD100K evaluation.

This module connects the evaluation results with the data analysis
to identify patterns in model failures and suggest improvements.
"""

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def cluster_failures_by_attribute(
    missed_detections: List[Dict],
    image_attributes: Dict[str, Dict],
) -> Dict[str, Dict[str, int]]:
    """Cluster model failures by image attributes.

    Args:
        missed_detections: List of missed detection entries.
        image_attributes: Dictionary mapping image names to their attributes.

    Returns:
        Dictionary mapping attributes to failure counts.
    """
    failures_by_attr = {
        "weather": defaultdict(int),
        "scene": defaultdict(int),
        "timeofday": defaultdict(int),
    }

    for miss in missed_detections:
        img_name = miss.get("image", "")
        attrs = image_attributes.get(img_name, {})

        for attr_name in failures_by_attr:
            attr_value = attrs.get(attr_name, "unknown")
            failures_by_attr[attr_name][attr_value] += 1

    return dict(failures_by_attr)


def cluster_failures_by_size(
    missed_detections: List[Dict],
) -> Dict[str, int]:
    """Cluster missed detections by object size.

    Args:
        missed_detections: List of missed detection entries with bbox info.

    Returns:
        Dictionary mapping size category to count.
    """
    size_failures = {"small": 0, "medium": 0, "large": 0}

    for miss in missed_detections:
        bbox = miss.get("bbox", [0, 0, 0, 0])
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        if area < 1024:
            size_failures["small"] += 1
        elif area < 9216:
            size_failures["medium"] += 1
        else:
            size_failures["large"] += 1

    return size_failures


def cluster_failures_by_position(
    missed_detections: List[Dict],
    img_width: int = 1280,
    img_height: int = 720,
    grid_size: int = 3,
) -> np.ndarray:
    """Cluster failures by spatial position in image.

    Divides the image into a grid and counts failures per cell.

    Args:
        missed_detections: List of missed detections with bbox info.
        img_width: Image width.
        img_height: Image height.
        grid_size: Grid dimensions (grid_size x grid_size).

    Returns:
        2D array with failure counts per grid cell.
    """
    grid = np.zeros((grid_size, grid_size), dtype=int)
    cell_w = img_width / grid_size
    cell_h = img_height / grid_size

    for miss in missed_detections:
        bbox = miss.get("bbox", [0, 0, 0, 0])
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        col = min(int(cx / cell_w), grid_size - 1)
        row = min(int(cy / cell_h), grid_size - 1)
        grid[row][col] += 1

    return grid


def analyze_class_confusion_patterns(
    classification_errors: List[Dict],
) -> pd.DataFrame:
    """Analyze patterns in classification errors.

    Args:
        classification_errors: List of classification error entries.

    Returns:
        DataFrame with confusion patterns.
    """
    confusion_pairs = Counter()
    for error in classification_errors:
        pair = (error["gt_class"], error["pred_class"])
        confusion_pairs[pair] += 1

    rows = []
    for (gt, pred), count in confusion_pairs.most_common():
        rows.append(
            {
                "ground_truth": gt,
                "predicted_as": pred,
                "count": count,
            }
        )

    return pd.DataFrame(rows)


def generate_improvement_suggestions(
    eval_results: Dict[str, Any],
    error_analysis: Dict[str, Any],
    data_analysis: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Generate improvement suggestions based on evaluation results.

    Connects evaluation findings with data analysis to provide
    actionable recommendations.

    Args:
        eval_results: Model evaluation results.
        error_analysis: Error analysis results.
        data_analysis: Optional data analysis results for context.

    Returns:
        List of improvement suggestion strings.
    """
    suggestions = []
    per_class = eval_results.get("per_class", {})
    summary = error_analysis.get("summary", {})

    # 1. Check for class imbalance issues
    aps = {cls: m["ap"] for cls, m in per_class.items()}
    if aps:
        worst_classes = sorted(aps.items(), key=lambda x: x[1])[:3]
        best_classes = sorted(aps.items(), key=lambda x: x[1], reverse=True)[:3]

        for cls, ap in worst_classes:
            if ap < 0.3:
                suggestions.append(
                    f"CLASS IMBALANCE: '{cls}' has low AP ({ap:.3f}). "
                    f"Consider data augmentation or oversampling for this class."
                )

    # 2. Check localization issues
    if summary.get("localization_errors", 0) > summary.get("classification_errors", 0):
        suggestions.append(
            "LOCALIZATION: More localization errors than classification errors. "
            "Consider: (1) Training with more IoU thresholds, "
            "(2) Using GIoU/DIoU loss, (3) Increasing input resolution."
        )

    # 3. Check missed detections
    if summary.get("missed_detections", 0) > 0:
        miss_rate = summary["missed_detections"]
        total_gt = sum(m.get("num_ground_truth", 0) for m in per_class.values())
        if total_gt > 0 and miss_rate / total_gt > 0.3:
            suggestions.append(
                "HIGH MISS RATE: >30% of ground truth objects are not detected. "
                "Consider: (1) Lowering confidence threshold, "
                "(2) Using larger model variant, (3) Multi-scale testing."
            )

    # 4. Check for small object issues
    small_classes = ["traffic light", "traffic sign"]
    for cls in small_classes:
        if cls in aps and aps[cls] < 0.4:
            suggestions.append(
                f"SMALL OBJECTS: '{cls}' detection is weak (AP={aps[cls]:.3f}). "
                f"Consider: (1) Higher input resolution (1280px), "
                f"(2) Adding more P3-level detection layers, "
                f"(3) Mosaic augmentation."
            )

    # 5. Check background FPs
    if summary.get("background_fps", 0) > summary.get("classification_errors", 0) * 2:
        suggestions.append(
            "BACKGROUND FPs: High number of false positive detections. "
            "Consider: (1) Increasing confidence threshold, "
            "(2) Hard negative mining, (3) More background augmentation."
        )

    # 6. Connect with data analysis if available
    if data_analysis:
        class_dist = data_analysis.get("class_distribution", {})
        if class_dist:
            for cls in worst_classes[:2]:
                cls_name = cls[0]
                if cls_name in class_dist and class_dist[cls_name] < 1000:
                    suggestions.append(
                        f"DATA SCARCITY: '{cls_name}' has few training samples "
                        f"({class_dist[cls_name]}). Collect more data or use "
                        f"synthetic augmentation (CutMix, copy-paste)."
                    )

    # 7. General suggestions
    suggestions.append(
        "GENERAL: Consider ensemble of multiple model scales (YOLO11s + YOLO11m) "
        "for improved accuracy at the cost of inference time."
    )
    suggestions.append(
        "TRAINING: Use longer training schedules (100+ epochs) with "
        "cosine LR scheduling and multi-scale training for best results."
    )

    return suggestions


def generate_full_analysis_report(
    eval_results: Dict[str, Any],
    error_analysis: Dict[str, Any],
    size_results: Dict[str, Dict],
    image_attributes: Optional[Dict[str, Dict]] = None,
) -> str:
    """Generate a comprehensive text analysis report.

    Args:
        eval_results: Evaluation results.
        error_analysis: Error analysis.
        size_results: Size-based results.
        image_attributes: Optional image attribute mapping.

    Returns:
        Formatted report string.
    """
    report = []
    report.append("=" * 70)
    report.append("BDD100K OBJECT DETECTION - COMPREHENSIVE EVALUATION REPORT")
    report.append("=" * 70)

    # Overall metrics
    report.append("\n## 1. OVERALL METRICS")
    report.append(f"   mAP@0.5: {eval_results.get('mAP', 0):.4f}")

    # Per-class breakdown
    report.append("\n## 2. PER-CLASS PERFORMANCE")
    report.append(f"   {'Class':<20} {'AP':<10} {'Precision':<12} {'Recall':<10}")
    report.append("   " + "-" * 52)

    for cls, metrics in sorted(
        eval_results.get("per_class", {}).items(),
        key=lambda x: x[1]["ap"],
        reverse=True,
    ):
        report.append(
            f"   {cls:<20} {metrics['ap']:<10.4f} "
            f"{metrics['precision']:<12.4f} {metrics['recall']:<10.4f}"
        )

    # Error analysis
    report.append("\n## 3. ERROR ANALYSIS")
    summary = error_analysis.get("summary", {})
    for error_type, count in summary.items():
        report.append(f"   {error_type}: {count}")

    # Size-based analysis
    report.append("\n## 4. PERFORMANCE BY OBJECT SIZE")
    for size, result in size_results.items():
        report.append(f"   {size}: mAP={result['mAP']:.4f}")

    # Improvement suggestions
    report.append("\n## 5. IMPROVEMENT SUGGESTIONS")
    suggestions = generate_improvement_suggestions(eval_results, error_analysis)
    for i, suggestion in enumerate(suggestions, 1):
        report.append(f"   {i}. {suggestion}")

    return "\n".join(report)
