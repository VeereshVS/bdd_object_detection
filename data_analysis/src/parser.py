"""BDD100K Dataset Parser.

This module provides data structures and parsing utilities for the BDD100K
dataset annotations in JSON format, focused on object detection tasks.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# BDD100K Object Detection classes
DETECTION_CLASSES = [
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

# Mapping for alternative class names
CLASS_NAME_MAP = {
    "person": "pedestrian",
    "light": "traffic light",
    "sign": "traffic sign",
}


@dataclass
class BoundingBox:
    """Represents a bounding box annotation."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Calculate bounding box width."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Calculate bounding box height."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Calculate bounding box area."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        """Calculate bounding box center coordinates."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio (width/height)."""
        if self.height == 0:
            return 0.0
        return self.width / self.height


@dataclass
class DetectionAnnotation:
    """Represents a single detection annotation."""

    category: str
    bbox: BoundingBox
    attributes: Dict = field(default_factory=dict)
    id: Optional[int] = None

    @property
    def is_occluded(self) -> bool:
        """Check if the object is occluded."""
        return self.attributes.get("occluded", False)

    @property
    def is_truncated(self) -> bool:
        """Check if the object is truncated."""
        return self.attributes.get("truncated", False)


@dataclass
class ImageAnnotation:
    """Represents all annotations for a single image."""

    image_name: str
    image_path: Optional[str] = None
    detections: List[DetectionAnnotation] = field(default_factory=list)
    attributes: Dict = field(default_factory=dict)
    timestamp: Optional[int] = None

    @property
    def weather(self) -> Optional[str]:
        """Get weather condition."""
        return self.attributes.get("weather", None)

    @property
    def scene(self) -> Optional[str]:
        """Get scene type."""
        return self.attributes.get("scene", None)

    @property
    def timeofday(self) -> Optional[str]:
        """Get time of day."""
        return self.attributes.get("timeofday", None)

    @property
    def num_objects(self) -> int:
        """Get total number of detection objects."""
        return len(self.detections)

    def get_objects_by_class(self, category: str) -> List[DetectionAnnotation]:
        """Get all detections of a specific class."""
        return [d for d in self.detections if d.category == category]


class BDD100KParser:
    """Parser for BDD100K dataset annotations.

    This parser reads the BDD100K JSON annotation files and provides
    structured access to the object detection annotations.

    Attributes:
        data_root: Root directory of the BDD100K dataset.
        annotations: Dictionary mapping split names to lists of ImageAnnotation.
    """

    def __init__(self, data_root: str):
        """Initialize the parser.

        Args:
            data_root: Root directory containing the BDD100K dataset.
        """
        self.data_root = Path(data_root)
        self.annotations: Dict[str, List[ImageAnnotation]] = {}

    def parse_labels(self, label_path: str, split: str = "train") -> List[ImageAnnotation]:
        """Parse a BDD100K label JSON file.

        Args:
            label_path: Path to the JSON label file.
            split: Dataset split name (train/val).

        Returns:
            List of ImageAnnotation objects.
        """
        label_path = Path(label_path)
        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")

        with open(label_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        annotations = []
        for item in raw_data:
            image_ann = self._parse_image_annotation(item, split)
            if image_ann.detections:  # Only keep images with detection annotations
                annotations.append(image_ann)

        self.annotations[split] = annotations
        return annotations

    def _parse_image_annotation(self, item: dict, split: str) -> ImageAnnotation:
        """Parse a single image annotation entry.

        Args:
            item: Raw annotation dictionary from JSON.
            split: Dataset split name.

        Returns:
            ImageAnnotation object.
        """
        image_name = item.get("name", "")
        image_path = str(
            self.data_root / "images" / "100k" / split / image_name
        )

        # Parse image-level attributes
        img_attributes = item.get("attributes", {})

        # Parse detection labels
        detections = []
        labels = item.get("labels", [])
        if labels:
            for label in labels:
                detection = self._parse_detection(label)
                if detection is not None:
                    detections.append(detection)

        return ImageAnnotation(
            image_name=image_name,
            image_path=image_path,
            detections=detections,
            attributes=img_attributes,
            timestamp=item.get("timestamp", None),
        )

    def _parse_detection(self, label: dict) -> Optional[DetectionAnnotation]:
        """Parse a single detection label.

        Args:
            label: Raw label dictionary.

        Returns:
            DetectionAnnotation object or None if not an object detection label.
        """
        category = label.get("category", "")

        # Normalize category name
        category = CLASS_NAME_MAP.get(category, category)

        # Only keep detection classes
        if category not in DETECTION_CLASSES:
            return None

        # Parse bounding box
        box2d = label.get("box2d", None)
        if box2d is None:
            return None

        bbox = BoundingBox(
            x1=float(box2d.get("x1", 0)),
            y1=float(box2d.get("y1", 0)),
            x2=float(box2d.get("x2", 0)),
            y2=float(box2d.get("y2", 0)),
        )

        # Parse attributes
        attributes = label.get("attributes", {})

        return DetectionAnnotation(
            category=category,
            bbox=bbox,
            attributes=attributes,
            id=label.get("id", None),
        )

    def get_class_distribution(self, split: str) -> Dict[str, int]:
        """Get the distribution of classes in a split.

        Args:
            split: Dataset split name.

        Returns:
            Dictionary mapping class names to counts.
        """
        if split not in self.annotations:
            raise ValueError(f"Split '{split}' not loaded. Call parse_labels first.")

        distribution = {cls: 0 for cls in DETECTION_CLASSES}
        for img_ann in self.annotations[split]:
            for det in img_ann.detections:
                if det.category in distribution:
                    distribution[det.category] += 1

        return distribution

    def get_images_per_class(self, split: str) -> Dict[str, int]:
        """Get number of images containing each class.

        Args:
            split: Dataset split name.

        Returns:
            Dictionary mapping class names to image counts.
        """
        if split not in self.annotations:
            raise ValueError(f"Split '{split}' not loaded. Call parse_labels first.")

        images_per_class = {cls: 0 for cls in DETECTION_CLASSES}
        for img_ann in self.annotations[split]:
            classes_in_image = set(d.category for d in img_ann.detections)
            for cls in classes_in_image:
                if cls in images_per_class:
                    images_per_class[cls] += 1

        return images_per_class

    def get_bbox_stats(self, split: str) -> Dict[str, Dict[str, float]]:
        """Get bounding box statistics per class.

        Args:
            split: Dataset split name.

        Returns:
            Dictionary with bbox statistics per class.
        """
        if split not in self.annotations:
            raise ValueError(f"Split '{split}' not loaded. Call parse_labels first.")

        stats = {}
        for cls in DETECTION_CLASSES:
            areas = []
            aspect_ratios = []
            widths = []
            heights = []

            for img_ann in self.annotations[split]:
                for det in img_ann.detections:
                    if det.category == cls:
                        areas.append(det.bbox.area)
                        aspect_ratios.append(det.bbox.aspect_ratio)
                        widths.append(det.bbox.width)
                        heights.append(det.bbox.height)

            if areas:
                stats[cls] = {
                    "count": len(areas),
                    "mean_area": sum(areas) / len(areas),
                    "min_area": min(areas),
                    "max_area": max(areas),
                    "mean_aspect_ratio": sum(aspect_ratios) / len(aspect_ratios),
                    "mean_width": sum(widths) / len(widths),
                    "mean_height": sum(heights) / len(heights),
                }

        return stats

    def get_attribute_distribution(self, split: str) -> Dict[str, Dict[str, int]]:
        """Get distribution of image-level attributes.

        Args:
            split: Dataset split name.

        Returns:
            Dictionary with attribute distributions.
        """
        if split not in self.annotations:
            raise ValueError(f"Split '{split}' not loaded. Call parse_labels first.")

        attributes = {"weather": {}, "scene": {}, "timeofday": {}}

        for img_ann in self.annotations[split]:
            for attr_name in attributes:
                value = img_ann.attributes.get(attr_name, "unknown")
                attributes[attr_name][value] = (
                    attributes[attr_name].get(value, 0) + 1
                )

        return attributes

    def get_occlusion_truncation_stats(
        self, split: str
    ) -> Dict[str, Dict[str, int]]:
        """Get occlusion and truncation statistics per class.

        Args:
            split: Dataset split name.

        Returns:
            Dictionary with occlusion/truncation counts per class.
        """
        if split not in self.annotations:
            raise ValueError(f"Split '{split}' not loaded. Call parse_labels first.")

        stats = {}
        for cls in DETECTION_CLASSES:
            occluded = 0
            truncated = 0
            total = 0

            for img_ann in self.annotations[split]:
                for det in img_ann.detections:
                    if det.category == cls:
                        total += 1
                        if det.is_occluded:
                            occluded += 1
                        if det.is_truncated:
                            truncated += 1

            if total > 0:
                stats[cls] = {
                    "total": total,
                    "occluded": occluded,
                    "truncated": truncated,
                    "occlusion_rate": occluded / total,
                    "truncation_rate": truncated / total,
                }

        return stats
