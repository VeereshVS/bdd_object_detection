"""Model Evaluation for BDD100K Object Detection.

This module provides comprehensive evaluation of the yolo11 model
on the BDD100K validation dataset, including mAP, precision, recall,
and per-class performance metrics.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two boxes.

    Args:
        box1: [x1, y1, x2, y2] format.
        box2: [x1, y1, x2, y2] format.

    Returns:
        IoU value between 0 and 1.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def calculate_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Calculate Average Precision using 11-point interpolation.

    Args:
        recalls: Array of recall values.
        precisions: Array of precision values.

    Returns:
        Average Precision value.
    """
    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0

    return ap


def evaluate_detections(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate object detection predictions against ground truth.

    Args:
        predictions: List of prediction dictionaries with format:
            {"image": str, "detections": [{"bbox", "class_name", "confidence"}]}
        ground_truths: List of ground truth dictionaries with format:
            {"image": str, "detections": [{"bbox", "class_name"}]}
        iou_threshold: IoU threshold for matching.
        classes: List of class names to evaluate.

    Returns:
        Dictionary with evaluation metrics.
    """
    if classes is None:
        classes = list(
            set(
                d["class_name"]
                for gt in ground_truths
                for d in gt.get("detections", [])
            )
        )

    # Organize by class
    all_predictions = defaultdict(list)
    all_ground_truths = defaultdict(lambda: defaultdict(list))
    n_gt_per_class = defaultdict(int)

    for gt in ground_truths:
        img_name = gt["image"]
        for det in gt.get("detections", []):
            cls_name = det["class_name"]
            all_ground_truths[cls_name][img_name].append(
                {
                    "bbox": det["bbox"],
                    "matched": False,
                }
            )
            n_gt_per_class[cls_name] += 1

    for pred in predictions:
        img_name = pred["image"]
        for det in pred.get("detections", []):
            cls_name = det["class_name"]
            all_predictions[cls_name].append(
                {
                    "image": img_name,
                    "bbox": det["bbox"],
                    "confidence": det["confidence"],
                }
            )

    # Calculate metrics per class
    results = {}
    all_aps = []

    for cls_name in classes:
        # Sort predictions by confidence
        preds = sorted(
            all_predictions[cls_name],
            key=lambda x: x["confidence"],
            reverse=True,
        )
        n_gt = n_gt_per_class[cls_name]

        if n_gt == 0:
            results[cls_name] = {
                "ap": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "num_predictions": len(preds),
                "num_ground_truth": 0,
                "tp": 0,
                "fp": len(preds),
            }
            continue

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))

        # Reset matched flags
        gt_matched = defaultdict(lambda: [False] * 100)
        for img_name, gt_boxes in all_ground_truths[cls_name].items():
            gt_matched[img_name] = [False] * len(gt_boxes)

        for pred_idx, pred in enumerate(preds):
            img_name = pred["image"]
            gt_boxes = all_ground_truths[cls_name].get(img_name, [])

            if not gt_boxes:
                fp[pred_idx] = 1
                continue

            # Find best matching ground truth
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_boxes):
                iou = calculate_iou(pred["bbox"], gt_box["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and not gt_matched[img_name][best_gt_idx]:
                tp[pred_idx] = 1
                gt_matched[img_name][best_gt_idx] = True
            else:
                fp[pred_idx] = 1

        # Calculate precision/recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / n_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        # Calculate AP
        ap = calculate_ap(recalls, precisions)
        all_aps.append(ap)

        results[cls_name] = {
            "ap": float(ap),
            "precision": float(precisions[-1]) if len(precisions) > 0 else 0.0,
            "recall": float(recalls[-1]) if len(recalls) > 0 else 0.0,
            "num_predictions": len(preds),
            "num_ground_truth": n_gt,
            "tp": int(np.sum(tp)),
            "fp": int(np.sum(fp)),
            "fn": n_gt - int(np.sum(tp)),
        }

    # Overall metrics
    mAP = np.mean(all_aps) if all_aps else 0.0

    return {
        "mAP": float(mAP),
        "per_class": results,
        "iou_threshold": iou_threshold,
        "num_classes": len(classes),
    }


def evaluate_at_multiple_ious(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_thresholds: List[float] = None,
    classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate at multiple IoU thresholds (COCO-style mAP@[.5:.95]).

    Args:
        predictions: List of prediction dictionaries.
        ground_truths: List of ground truth dictionaries.
        iou_thresholds: List of IoU thresholds to evaluate at.
        classes: List of class names.

    Returns:
        Dictionary with evaluation metrics at each threshold.
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05).tolist()

    results = {}
    all_maps = []

    for iou_thresh in iou_thresholds:
        result = evaluate_detections(
            predictions, ground_truths, iou_threshold=iou_thresh, classes=classes
        )
        results[f"iou_{iou_thresh:.2f}"] = result
        all_maps.append(result["mAP"])

    results["mAP_50_95"] = float(np.mean(all_maps))
    results["mAP_50"] = results.get("iou_0.50", {}).get("mAP", 0.0)
    results["mAP_75"] = results.get("iou_0.75", {}).get("mAP", 0.0)

    return results


def analyze_errors(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Analyze detection errors by category.

    Categorizes errors into:
    - Classification errors: correct localization, wrong class
    - Localization errors: correct class, low IoU
    - Duplicate detections: multiple detections for one GT
    - Background false positives: detection where no GT exists
    - Missed detections: GT objects not detected

    Args:
        predictions: Prediction results.
        ground_truths: Ground truth annotations.
        iou_threshold: IoU threshold for matching.

    Returns:
        Dictionary with error analysis.
    """
    errors = {
        "classification_errors": [],
        "localization_errors": [],
        "duplicate_detections": [],
        "background_fps": [],
        "missed_detections": [],
    }

    for pred, gt in zip(predictions, ground_truths):
        if pred["image"] != gt["image"]:
            continue

        pred_dets = pred.get("detections", [])
        gt_dets = gt.get("detections", [])
        gt_matched = [False] * len(gt_dets)

        for p_det in sorted(pred_dets, key=lambda x: x["confidence"], reverse=True):
            best_iou = 0.0
            best_gt_idx = -1

            for g_idx, g_det in enumerate(gt_dets):
                iou = calculate_iou(p_det["bbox"], g_det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= iou_threshold:
                if gt_matched[best_gt_idx]:
                    errors["duplicate_detections"].append(
                        {
                            "image": pred["image"],
                            "pred_class": p_det["class_name"],
                            "confidence": p_det["confidence"],
                        }
                    )
                elif p_det["class_name"] != gt_dets[best_gt_idx]["class_name"]:
                    errors["classification_errors"].append(
                        {
                            "image": pred["image"],
                            "pred_class": p_det["class_name"],
                            "gt_class": gt_dets[best_gt_idx]["class_name"],
                            "confidence": p_det["confidence"],
                            "iou": best_iou,
                        }
                    )
                    gt_matched[best_gt_idx] = True
                else:
                    gt_matched[best_gt_idx] = True
            elif best_iou >= 0.1:
                errors["localization_errors"].append(
                    {
                        "image": pred["image"],
                        "pred_class": p_det["class_name"],
                        "confidence": p_det["confidence"],
                        "iou": best_iou,
                    }
                )
            else:
                errors["background_fps"].append(
                    {
                        "image": pred["image"],
                        "pred_class": p_det["class_name"],
                        "confidence": p_det["confidence"],
                    }
                )

        # Find missed detections
        for g_idx, matched in enumerate(gt_matched):
            if not matched:
                errors["missed_detections"].append(
                    {
                        "image": gt["image"],
                        "gt_class": gt_dets[g_idx]["class_name"],
                        "bbox": gt_dets[g_idx]["bbox"],
                    }
                )

    # Summary
    errors["summary"] = {
        "classification_errors": len(errors["classification_errors"]),
        "localization_errors": len(errors["localization_errors"]),
        "duplicate_detections": len(errors["duplicate_detections"]),
        "background_fps": len(errors["background_fps"]),
        "missed_detections": len(errors["missed_detections"]),
    }

    return errors


def evaluate_by_object_size(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict[str, Dict]:
    """Evaluate detection performance by object size.

    Size categories (following COCO convention):
    - Small: area < 32² = 1024 pixels
    - Medium: 32² <= area < 96² = 9216 pixels
    - Large: area >= 96²

    Args:
        predictions: Prediction results.
        ground_truths: Ground truth annotations.
        iou_threshold: IoU threshold.

    Returns:
        Dictionary with metrics for each size category.
    """
    size_categories = {
        "small": (0, 1024),
        "medium": (1024, 9216),
        "large": (9216, float("inf")),
    }

    results = {}

    for size_name, (min_area, max_area) in size_categories.items():
        # Filter ground truths by size
        filtered_gt = []
        for gt in ground_truths:
            filtered_dets = []
            for det in gt.get("detections", []):
                bbox = det["bbox"]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if min_area <= area < max_area:
                    filtered_dets.append(det)
            filtered_gt.append(
                {"image": gt["image"], "detections": filtered_dets}
            )

        # Filter predictions by size
        filtered_preds = []
        for pred in predictions:
            filtered_dets = []
            for det in pred.get("detections", []):
                bbox = det["bbox"]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if min_area <= area < max_area:
                    filtered_dets.append(det)
            filtered_preds.append(
                {"image": pred["image"], "detections": filtered_dets}
            )

        # Evaluate
        eval_result = evaluate_detections(
            filtered_preds, filtered_gt, iou_threshold=iou_threshold
        )
        results[size_name] = eval_result

    return results