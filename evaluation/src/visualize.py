"""Visualization tools for model evaluation results.

This module provides both quantitative (charts, plots) and qualitative
(image-level) visualization of model performance on BDD100K.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_precision_recall_curve(
    predictions: List[Dict],
    ground_truths: List[Dict],
    class_name: str,
    iou_threshold: float = 0.5,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot precision-recall curve for a specific class.

    Args:
        predictions: List of prediction dictionaries.
        ground_truths: List of ground truth dictionaries.
        class_name: Class to plot curve for.
        iou_threshold: IoU threshold for matching.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    from .evaluate import calculate_iou

    # Collect predictions for this class
    all_preds = []
    gt_per_image = defaultdict(list)
    total_gt = 0

    for gt in ground_truths:
        for det in gt.get("detections", []):
            if det["class_name"] == class_name:
                gt_per_image[gt["image"]].append(det["bbox"])
                total_gt += 1

    for pred in predictions:
        for det in pred.get("detections", []):
            if det["class_name"] == class_name:
                all_preds.append(
                    {
                        "image": pred["image"],
                        "bbox": det["bbox"],
                        "confidence": det["confidence"],
                    }
                )

    # Sort by confidence
    all_preds.sort(key=lambda x: x["confidence"], reverse=True)

    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))
    matched = defaultdict(lambda: set())

    for i, pred in enumerate(all_preds):
        gt_boxes = gt_per_image.get(pred["image"], [])
        best_iou = 0.0
        best_idx = -1

        for idx, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred["bbox"], gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou >= iou_threshold and best_idx not in matched[pred["image"]]:
            tp[i] = 1
            matched[pred["image"]].add(best_idx)
        else:
            fp[i] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / total_gt if total_gt > 0 else tp_cumsum
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recalls, precisions, "b-", linewidth=2)
    ax.fill_between(recalls, precisions, alpha=0.2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve: {class_name} (IoU={iou_threshold})")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_map_per_class(
    eval_results: Dict[str, Any], save_path: Optional[str] = None
) -> plt.Figure:
    """Plot mAP bar chart per class.

    Args:
        eval_results: Evaluation results dictionary.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    per_class = eval_results.get("per_class", {})
    classes = list(per_class.keys())
    aps = [per_class[c]["ap"] for c in classes]

    # Sort by AP
    sorted_indices = np.argsort(aps)[::-1]
    classes = [classes[i] for i in sorted_indices]
    aps = [aps[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.RdYlGn(np.array(aps))
    bars = ax.barh(classes, aps, color=colors)

    ax.set_xlabel("Average Precision (AP)")
    ax.set_title(
        f"Per-Class AP @ IoU={eval_results.get('iou_threshold', 0.5):.2f} | "
        f"mAP={eval_results.get('mAP', 0):.4f}"
    )
    ax.set_xlim([0, 1])
    ax.axvline(x=eval_results.get("mAP", 0), color="red", linestyle="--", alpha=0.7)

    # Add value labels
    for bar, ap in zip(bars, aps):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{ap:.3f}",
            va="center",
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    predictions: List[Dict],
    ground_truths: List[Dict],
    classes: List[str],
    iou_threshold: float = 0.5,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot confusion matrix for object detection.

    Args:
        predictions: Prediction results.
        ground_truths: Ground truth annotations.
        classes: List of class names.
        iou_threshold: IoU threshold for matching.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    from .evaluate import calculate_iou

    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Add background class
    matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=int)

    for pred, gt in zip(predictions, ground_truths):
        gt_dets = gt.get("detections", [])
        pred_dets = pred.get("detections", [])
        gt_matched = [False] * len(gt_dets)

        for p_det in sorted(pred_dets, key=lambda x: x["confidence"], reverse=True):
            best_iou = 0.0
            best_gt_idx = -1

            for g_idx, g_det in enumerate(gt_dets):
                iou = calculate_iou(p_det["bbox"], g_det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            pred_cls = class_to_idx.get(p_det["class_name"], n_classes)

            if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
                gt_cls = class_to_idx.get(
                    gt_dets[best_gt_idx]["class_name"], n_classes
                )
                matrix[gt_cls][pred_cls] += 1
                gt_matched[best_gt_idx] = True
            else:
                # False positive (background -> predicted class)
                matrix[n_classes][pred_cls] += 1

        # Missed detections
        for g_idx, matched in enumerate(gt_matched):
            if not matched:
                gt_cls = class_to_idx.get(gt_dets[g_idx]["class_name"], n_classes)
                matrix[gt_cls][n_classes] += 1

    labels = classes + ["background"]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Detection Confusion Matrix")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_error_analysis(
    error_results: Dict[str, Any], save_path: Optional[str] = None
) -> plt.Figure:
    """Plot error analysis breakdown.

    Args:
        error_results: Error analysis results.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    summary = error_results["summary"]
    categories = list(summary.keys())
    values = list(summary.values())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pie chart
    colors = ["#ff6b6b", "#ffa500", "#4ecdc4", "#45b7d1", "#96ceb4"]
    axes[0].pie(values, labels=categories, colors=colors, autopct="%1.1f%%")
    axes[0].set_title("Error Distribution")

    # Bar chart
    axes[1].barh(categories, values, color=colors)
    axes[1].set_xlabel("Count")
    axes[1].set_title("Error Counts by Category")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_performance_by_size(
    size_results: Dict[str, Dict], save_path: Optional[str] = None
) -> plt.Figure:
    """Plot performance breakdown by object size.

    Args:
        size_results: Results from evaluate_by_object_size.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    sizes = list(size_results.keys())
    maps = [size_results[s]["mAP"] for s in sizes]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(sizes, maps, color=["#ff6b6b", "#4ecdc4", "#45b7d1"])

    ax.set_xlabel("Object Size")
    ax.set_ylabel("mAP")
    ax.set_title("Detection Performance by Object Size")
    ax.set_ylim([0, 1])

    for bar, m in zip(bars, maps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{m:.3f}",
            ha="center",
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_predictions_vs_gt(
    image_path: str,
    predictions: List[Dict],
    ground_truths: List[Dict],
    save_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Visualize predictions vs ground truth side by side.

    Args:
        image_path: Path to the image.
        predictions: Prediction detections for this image.
        ground_truths: Ground truth detections for this image.
        save_path: Path to save the visualization.

    Returns:
        Combined visualization image or None.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Color scheme
    gt_color = (0, 255, 0)  # Green for GT
    pred_color = (0, 0, 255)  # Red for predictions

    # Create two copies
    img_gt = img.copy()
    img_pred = img.copy()

    # Draw ground truth
    for det in ground_truths:
        x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
        cv2.rectangle(img_gt, (x1, y1), (x2, y2), gt_color, 2)
        label = det["class_name"]
        cv2.putText(
            img_gt, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, gt_color, 1,
        )

    # Draw predictions
    for det in predictions:
        x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
        cv2.rectangle(img_pred, (x1, y1), (x2, y2), pred_color, 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.putText(
            img_pred, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, pred_color, 1,
        )

    # Add titles
    cv2.putText(img_gt, "Ground Truth", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, gt_color, 2)
    cv2.putText(img_pred, "Predictions", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, pred_color, 2)

    # Concatenate side by side
    combined = np.hstack([img_gt, img_pred])

    if save_path:
        cv2.imwrite(save_path, combined)

    return combined


def plot_performance_by_attribute(
    results_by_attr: Dict[str, Dict[str, float]],
    attribute_name: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot model performance broken down by image attribute.

    Args:
        results_by_attr: Dictionary mapping attribute values to mAP scores.
        attribute_name: Name of the attribute (weather, scene, timeofday).
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    attrs = list(results_by_attr.keys())
    maps = [results_by_attr[a] for a in attrs]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(attrs, maps, color=plt.cm.Set3(np.linspace(0, 1, len(attrs))))

    ax.set_xlabel(attribute_name.capitalize())
    ax.set_ylabel("mAP")
    ax.set_title(f"Detection Performance by {attribute_name.capitalize()}")
    ax.set_ylim([0, 1])
    ax.tick_params(axis="x", rotation=45)

    for bar, m in zip(bars, maps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{m:.3f}",
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_evaluation_report(
    eval_results: Dict[str, Any],
    error_analysis: Dict[str, Any],
    size_results: Dict[str, Dict],
    output_dir: str,
) -> str:
    """Generate a comprehensive evaluation report with all visualizations.

    Args:
        eval_results: Main evaluation results.
        error_analysis: Error analysis results.
        size_results: Size-based evaluation results.
        output_dir: Directory to save report and plots.

    Returns:
        Path to the generated report.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plots_dir = output_path / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Generate all plots
    plot_map_per_class(eval_results, str(plots_dir / "map_per_class.png"))
    plot_error_analysis(error_analysis, str(plots_dir / "error_analysis.png"))
    plot_performance_by_size(size_results, str(plots_dir / "performance_by_size.png"))

    # Generate text report
    report_lines = [
        "# BDD100K Object Detection - Evaluation Report\n",
        f"## Overall Metrics",
        f"- **mAP@0.5**: {eval_results.get('mAP', 0):.4f}",
        f"- **IoU Threshold**: {eval_results.get('iou_threshold', 0.5)}",
        "",
        "## Per-Class Performance",
    ]

    for cls, metrics in eval_results.get("per_class", {}).items():
        report_lines.append(
            f"- **{cls}**: AP={metrics['ap']:.4f}, "
            f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}"
        )

    report_lines.extend([
        "",
        "## Error Analysis",
        f"- Classification Errors: {error_analysis['summary']['classification_errors']}",
        f"- Localization Errors: {error_analysis['summary']['localization_errors']}",
        f"- Duplicate Detections: {error_analysis['summary']['duplicate_detections']}",
        f"- Background FPs: {error_analysis['summary']['background_fps']}",
        f"- Missed Detections: {error_analysis['summary']['missed_detections']}",
        "",
        "## Performance by Object Size",
    ])

    for size, result in size_results.items():
        report_lines.append(f"- **{size}**: mAP={result['mAP']:.4f}")

    report_text = "\n".join(report_lines)
    report_path = str(output_path / "evaluation_report.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_path
