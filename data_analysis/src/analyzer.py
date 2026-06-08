"""BDD100K Dataset Analyzer.

This module provides comprehensive analysis functions for the BDD100K
object detection dataset including distribution analysis, anomaly detection,
and pattern identification.
"""

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .parser import (
    BDD100KParser,
    DETECTION_CLASSES,
    ImageAnnotation,
)

logger = logging.getLogger(__name__)


class BDD100KAnalyzer:
    """Comprehensive analyzer for BDD100K object detection dataset.

    Provides methods for analyzing class distributions, identifying anomalies,
    and extracting patterns from the dataset annotations.

    Attributes:
        parser: BDD100KParser instance with loaded annotations.
    """

    def __init__(self, parser: BDD100KParser):
        """Initialize the analyzer.

        Args:
            parser: BDD100KParser instance with parsed annotations.
        """
        self.parser = parser

    def analyze_class_distribution(self, split: str) -> pd.DataFrame:
        """Analyze class distribution for a given split.

        Args:
            split: Dataset split name (train/val).

        Returns:
            DataFrame with class distribution statistics.
        """
        distribution = self.parser.get_class_distribution(split)
        images_per_class = self.parser.get_images_per_class(split)
        total_instances = sum(distribution.values())
        total_images = len(self.parser.annotations[split])

        rows = []
        for cls in DETECTION_CLASSES:
            count = distribution[cls]
            img_count = images_per_class[cls]
            rows.append(
                {
                    "class": cls,
                    "instance_count": count,
                    "percentage": (count / total_instances * 100)
                    if total_instances > 0
                    else 0,
                    "images_containing": img_count,
                    "image_percentage": (img_count / total_images * 100)
                    if total_images > 0
                    else 0,
                    "avg_instances_per_image": (count / img_count)
                    if img_count > 0
                    else 0,
                }
            )

        df = pd.DataFrame(rows)
        df = df.sort_values("instance_count", ascending=False).reset_index(drop=True)
        return df

    def analyze_train_val_split(self) -> pd.DataFrame:
        """Compare training and validation split distributions.

        Returns:
            DataFrame comparing train and val distributions.
        """
        train_dist = self.parser.get_class_distribution("train")
        val_dist = self.parser.get_class_distribution("val")

        train_total = sum(train_dist.values())
        val_total = sum(val_dist.values())

        rows = []
        for cls in DETECTION_CLASSES:
            train_count = train_dist[cls]
            val_count = val_dist[cls]
            train_pct = (train_count / train_total * 100) if train_total > 0 else 0
            val_pct = (val_count / val_total * 100) if val_total > 0 else 0

            rows.append(
                {
                    "class": cls,
                    "train_count": train_count,
                    "train_percentage": train_pct,
                    "val_count": val_count,
                    "val_percentage": val_pct,
                    "distribution_diff": abs(train_pct - val_pct),
                    "train_val_ratio": (train_count / val_count)
                    if val_count > 0
                    else float("inf"),
                }
            )

        return pd.DataFrame(rows)

    def detect_anomalies(self, split: str) -> Dict[str, Any]:
        """Detect anomalies in the dataset.

        Identifies:
        - Images with unusually high/low number of objects
        - Extremely small or large bounding boxes
        - Objects with invalid coordinates
        - Class co-occurrence patterns

        Args:
            split: Dataset split name.

        Returns:
            Dictionary containing detected anomalies.
        """
        annotations = self.parser.annotations[split]
        anomalies = {
            "high_density_images": [],
            "empty_images": [],
            "tiny_boxes": [],
            "huge_boxes": [],
            "invalid_boxes": [],
            "extreme_aspect_ratios": [],
        }

        # Calculate statistics for thresholds
        objects_per_image = [ann.num_objects for ann in annotations]
        mean_objects = np.mean(objects_per_image)
        std_objects = np.std(objects_per_image)
        threshold_high = mean_objects + 3 * std_objects

        # Image dimensions (BDD100K is 1280x720)
        img_area = 1280 * 720
        tiny_threshold = img_area * 0.0001  # Less than 0.01% of image
        huge_threshold = img_area * 0.5  # More than 50% of image

        for ann in annotations:
            # High density images
            if ann.num_objects > threshold_high:
                anomalies["high_density_images"].append(
                    {
                        "image": ann.image_name,
                        "num_objects": ann.num_objects,
                    }
                )

            for det in ann.detections:
                bbox = det.bbox

                # Invalid boxes
                if bbox.width <= 0 or bbox.height <= 0:
                    anomalies["invalid_boxes"].append(
                        {
                            "image": ann.image_name,
                            "category": det.category,
                            "bbox": (bbox.x1, bbox.y1, bbox.x2, bbox.y2),
                        }
                    )
                    continue

                # Tiny boxes
                if bbox.area < tiny_threshold:
                    anomalies["tiny_boxes"].append(
                        {
                            "image": ann.image_name,
                            "category": det.category,
                            "area": bbox.area,
                        }
                    )

                # Huge boxes
                if bbox.area > huge_threshold:
                    anomalies["huge_boxes"].append(
                        {
                            "image": ann.image_name,
                            "category": det.category,
                            "area": bbox.area,
                        }
                    )

                # Extreme aspect ratios
                ar = bbox.aspect_ratio
                if ar > 10 or ar < 0.1:
                    anomalies["extreme_aspect_ratios"].append(
                        {
                            "image": ann.image_name,
                            "category": det.category,
                            "aspect_ratio": ar,
                        }
                    )

        # Summary
        anomalies["summary"] = {
            "total_images": len(annotations),
            "high_density_count": len(anomalies["high_density_images"]),
            "tiny_boxes_count": len(anomalies["tiny_boxes"]),
            "huge_boxes_count": len(anomalies["huge_boxes"]),
            "invalid_boxes_count": len(anomalies["invalid_boxes"]),
            "extreme_ar_count": len(anomalies["extreme_aspect_ratios"]),
        }

        return anomalies

    def analyze_class_cooccurrence(self, split: str) -> pd.DataFrame:
        """Analyze which classes frequently appear together.

        Args:
            split: Dataset split name.

        Returns:
            DataFrame with co-occurrence matrix.
        """
        annotations = self.parser.annotations[split]
        cooccurrence = np.zeros(
            (len(DETECTION_CLASSES), len(DETECTION_CLASSES)), dtype=int
        )
        class_to_idx = {cls: i for i, cls in enumerate(DETECTION_CLASSES)}

        for ann in annotations:
            classes_in_image = list(set(d.category for d in ann.detections))
            for i, cls1 in enumerate(classes_in_image):
                if cls1 not in class_to_idx:
                    continue
                for cls2 in classes_in_image[i:]:
                    if cls2 not in class_to_idx:
                        continue
                    idx1 = class_to_idx[cls1]
                    idx2 = class_to_idx[cls2]
                    cooccurrence[idx1][idx2] += 1
                    if idx1 != idx2:
                        cooccurrence[idx2][idx1] += 1

        return pd.DataFrame(
            cooccurrence, index=DETECTION_CLASSES, columns=DETECTION_CLASSES
        )

    def analyze_spatial_distribution(self, split: str) -> Dict[str, Any]:
        """Analyze spatial distribution of objects in images.

        Args:
            split: Dataset split name.

        Returns:
            Dictionary with spatial statistics per class.
        """
        annotations = self.parser.annotations[split]
        spatial_data = defaultdict(lambda: {"cx": [], "cy": [], "w": [], "h": []})

        # BDD100K image dimensions
        img_w, img_h = 1280, 720

        for ann in annotations:
            for det in ann.detections:
                cx, cy = det.bbox.center
                # Normalize to [0, 1]
                spatial_data[det.category]["cx"].append(cx / img_w)
                spatial_data[det.category]["cy"].append(cy / img_h)
                spatial_data[det.category]["w"].append(det.bbox.width / img_w)
                spatial_data[det.category]["h"].append(det.bbox.height / img_h)

        results = {}
        for cls in DETECTION_CLASSES:
            if cls in spatial_data and spatial_data[cls]["cx"]:
                data = spatial_data[cls]
                results[cls] = {
                    "mean_cx": np.mean(data["cx"]),
                    "mean_cy": np.mean(data["cy"]),
                    "std_cx": np.std(data["cx"]),
                    "std_cy": np.std(data["cy"]),
                    "mean_w": np.mean(data["w"]),
                    "mean_h": np.mean(data["h"]),
                    "std_w": np.std(data["w"]),
                    "std_h": np.std(data["h"]),
                }

        return results

    def analyze_by_attributes(self, split: str) -> Dict[str, pd.DataFrame]:
        """Analyze detection distribution by image attributes.

        Args:
            split: Dataset split name.

        Returns:
            Dictionary with DataFrames for each attribute type.
        """
        annotations = self.parser.annotations[split]
        results = {}

        for attr in ["weather", "scene", "timeofday"]:
            data = defaultdict(lambda: defaultdict(int))
            for ann in annotations:
                attr_value = ann.attributes.get(attr, "unknown")
                for det in ann.detections:
                    data[attr_value][det.category] += 1

            df = pd.DataFrame(data).T.fillna(0).astype(int)
            # Ensure all classes are present
            for cls in DETECTION_CLASSES:
                if cls not in df.columns:
                    df[cls] = 0
            df = df[DETECTION_CLASSES]
            results[attr] = df

        return results

    def find_interesting_samples(
        self, split: str, top_k: int = 10
    ) -> Dict[str, List[Dict]]:
        """Find interesting/unique samples in the dataset.

        Identifies:
        - Images with the most objects
        - Images with rare class combinations
        - Images with objects of extreme sizes
        - Images with single class dominance

        Args:
            split: Dataset split name.
            top_k: Number of samples to return per category.

        Returns:
            Dictionary of interesting samples by category.
        """
        annotations = self.parser.annotations[split]
        interesting = {
            "most_crowded": [],
            "most_diverse": [],
            "single_class_dominant": [],
            "rare_classes": [],
            "tiny_objects": [],
            "large_objects": [],
        }

        # Most crowded images
        sorted_by_count = sorted(
            annotations, key=lambda x: x.num_objects, reverse=True
        )
        for ann in sorted_by_count[:top_k]:
            interesting["most_crowded"].append(
                {
                    "image": ann.image_name,
                    "num_objects": ann.num_objects,
                    "classes": Counter(d.category for d in ann.detections),
                }
            )

        # Most diverse (most unique classes)
        sorted_by_diversity = sorted(
            annotations,
            key=lambda x: len(set(d.category for d in x.detections)),
            reverse=True,
        )
        for ann in sorted_by_diversity[:top_k]:
            classes = set(d.category for d in ann.detections)
            interesting["most_diverse"].append(
                {
                    "image": ann.image_name,
                    "num_classes": len(classes),
                    "classes": list(classes),
                }
            )

        # Single class dominant (one class makes up >80% of detections)
        for ann in annotations:
            if ann.num_objects >= 5:
                class_counts = Counter(d.category for d in ann.detections)
                most_common_cls, most_common_count = class_counts.most_common(1)[0]
                if most_common_count / ann.num_objects > 0.8:
                    interesting["single_class_dominant"].append(
                        {
                            "image": ann.image_name,
                            "dominant_class": most_common_cls,
                            "dominance_ratio": most_common_count / ann.num_objects,
                            "total_objects": ann.num_objects,
                        }
                    )

        # Keep only top_k
        interesting["single_class_dominant"] = sorted(
            interesting["single_class_dominant"],
            key=lambda x: x["dominance_ratio"],
            reverse=True,
        )[:top_k]

        # Tiny objects (smallest bounding boxes)
        all_tiny = []
        for ann in annotations:
            for det in ann.detections:
                if det.bbox.area > 0:
                    all_tiny.append(
                        {
                            "image": ann.image_name,
                            "category": det.category,
                            "area": det.bbox.area,
                            "width": det.bbox.width,
                            "height": det.bbox.height,
                        }
                    )
        all_tiny.sort(key=lambda x: x["area"])
        interesting["tiny_objects"] = all_tiny[:top_k]

        # Large objects (largest bounding boxes)
        all_tiny.sort(key=lambda x: x["area"], reverse=True)
        interesting["large_objects"] = all_tiny[:top_k]

        # Rare classes (least common detection categories)
        # Note: "train" here refers to railway/rail vehicles, NOT the data split
        distribution = self.parser.get_class_distribution(split)
        rare_classes = sorted(distribution.items(), key=lambda x: x[1])[:3]
        for cls, count in rare_classes:
            for ann in annotations:
                if any(d.category == cls for d in ann.detections):
                    interesting["rare_classes"].append(
                        {
                            "image": ann.image_name,
                            "rare_class": f"{cls} (total instances: {count})",
                            "category": cls,
                            "all_classes": Counter(
                                d.category for d in ann.detections
                            ),
                        }
                    )
                    if len(interesting["rare_classes"]) >= top_k:
                        break
            if len(interesting["rare_classes"]) >= top_k:
                break

        return interesting

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a comprehensive summary report of the dataset.

        Returns:
            Dictionary containing all analysis results.
        """
        report = {}

        for split in self.parser.annotations:
            split_report = {
                "total_images": len(self.parser.annotations[split]),
                "class_distribution": self.analyze_class_distribution(split).to_dict(
                    orient="records"
                ),
                "bbox_stats": self.parser.get_bbox_stats(split),
                "attribute_distribution": self.parser.get_attribute_distribution(split),
                "occlusion_stats": self.parser.get_occlusion_truncation_stats(split),
                "spatial_distribution": self.analyze_spatial_distribution(split),
            }
            report[split] = split_report

        if "train" in self.parser.annotations and "val" in self.parser.annotations:
            report["train_val_comparison"] = self.analyze_train_val_split().to_dict(
                orient="records"
            )

        return report
