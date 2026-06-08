"""BDD100K Dataset Visualizer.

This module provides visualization utilities for the BDD100K dataset analysis,
including charts, plots, and a Streamlit-based interactive dashboard.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .parser import BDD100KParser, DETECTION_CLASSES, ImageAnnotation


# Color palette for detection classes
CLASS_COLORS = {
    "pedestrian": (255, 0, 0),
    "rider": (255, 128, 0),
    "car": (0, 255, 0),
    "truck": (0, 0, 255),
    "bus": (255, 255, 0),
    "train": (255, 0, 255),
    "motorcycle": (0, 255, 255),
    "bicycle": (128, 0, 255),
    "traffic light": (128, 255, 0),
    "traffic sign": (0, 128, 255),
}


def plot_class_distribution(
    distribution_df: pd.DataFrame,
    title: str = "Class Distribution",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot class distribution as a bar chart.

    Args:
        distribution_df: DataFrame with class distribution data.
        title: Plot title.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Instance count
    sns.barplot(
        data=distribution_df,
        x="class",
        y="instance_count",
        ax=axes[0],
        palette="viridis",
    )
    axes[0].set_title(f"{title} - Instance Count")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=45)

    # Images containing class
    sns.barplot(
        data=distribution_df,
        x="class",
        y="images_containing",
        ax=axes[1],
        palette="magma",
    )
    axes[1].set_title(f"{title} - Images Containing Class")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Image Count")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_train_val_comparison(
    comparison_df: pd.DataFrame, save_path: Optional[str] = None
) -> plt.Figure:
    """Plot train vs validation distribution comparison.

    Args:
        comparison_df: DataFrame with train/val comparison data.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(comparison_df))
    width = 0.35

    ax.bar(
        x - width / 2,
        comparison_df["train_percentage"],
        width,
        label="Train",
        color="steelblue",
    )
    ax.bar(
        x + width / 2,
        comparison_df["val_percentage"],
        width,
        label="Validation",
        color="coral",
    )

    ax.set_xlabel("Class")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Train vs Validation Class Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["class"], rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_bbox_statistics(
    bbox_stats: Dict[str, Dict[str, float]], save_path: Optional[str] = None
) -> plt.Figure:
    """Plot bounding box statistics.

    Args:
        bbox_stats: Dictionary with bbox statistics per class.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    classes = list(bbox_stats.keys())
    mean_areas = [bbox_stats[c]["mean_area"] for c in classes]
    mean_widths = [bbox_stats[c]["mean_width"] for c in classes]
    mean_heights = [bbox_stats[c]["mean_height"] for c in classes]
    mean_ar = [bbox_stats[c]["mean_aspect_ratio"] for c in classes]

    # Mean area
    axes[0, 0].barh(classes, mean_areas, color="teal")
    axes[0, 0].set_title("Mean Bounding Box Area")
    axes[0, 0].set_xlabel("Pixels²")

    # Mean width
    axes[0, 1].barh(classes, mean_widths, color="orange")
    axes[0, 1].set_title("Mean Bounding Box Width")
    axes[0, 1].set_xlabel("Pixels")

    # Mean height
    axes[1, 0].barh(classes, mean_heights, color="purple")
    axes[1, 0].set_title("Mean Bounding Box Height")
    axes[1, 0].set_xlabel("Pixels")

    # Mean aspect ratio
    axes[1, 1].barh(classes, mean_ar, color="green")
    axes[1, 1].set_title("Mean Aspect Ratio (W/H)")
    axes[1, 1].set_xlabel("Ratio")
    axes[1, 1].axvline(x=1.0, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_cooccurrence_matrix(
    cooccurrence_df: pd.DataFrame, save_path: Optional[str] = None
) -> plt.Figure:
    """Plot class co-occurrence heatmap.

    Args:
        cooccurrence_df: DataFrame with co-occurrence counts.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        cooccurrence_df,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        ax=ax,
        square=True,
    )
    ax.set_title("Class Co-occurrence Matrix")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_attribute_distribution(
    attribute_data: Dict[str, Dict[str, int]], save_path: Optional[str] = None
) -> plt.Figure:
    """Plot image attribute distributions.

    Args:
        attribute_data: Dictionary with attribute distributions.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (attr_name, attr_dict) in enumerate(attribute_data.items()):
        labels = list(attr_dict.keys())
        values = list(attr_dict.values())
        axes[idx].pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        axes[idx].set_title(f"Distribution by {attr_name.capitalize()}")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_spatial_heatmap(
    annotations: List[ImageAnnotation],
    class_name: str,
    img_width: int = 1280,
    img_height: int = 720,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot spatial heatmap of object locations.

    Args:
        annotations: List of image annotations.
        class_name: Class to visualize.
        img_width: Image width.
        img_height: Image height.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    heatmap = np.zeros((img_height, img_width), dtype=np.float32)

    for ann in annotations:
        for det in ann.detections:
            if det.category == class_name:
                x1 = int(max(0, det.bbox.x1))
                y1 = int(max(0, det.bbox.y1))
                x2 = int(min(img_width, det.bbox.x2))
                y2 = int(min(img_height, det.bbox.y2))
                heatmap[y1:y2, x1:x2] += 1

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(heatmap, cmap="hot", interpolation="gaussian")
    ax.set_title(f"Spatial Distribution Heatmap - {class_name}")
    plt.colorbar(im, ax=ax, label="Object density")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_sample(
    image_path: str,
    annotation: ImageAnnotation,
    save_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Visualize a single sample with bounding box annotations.

    Args:
        image_path: Path to the image file.
        annotation: ImageAnnotation for this image.
        save_path: Path to save the visualization.

    Returns:
        Annotated image as numpy array, or None if image not found.
    """
    if not os.path.exists(image_path):
        return None

    img = cv2.imread(image_path)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for det in annotation.detections:
        color = CLASS_COLORS.get(det.category, (255, 255, 255))
        x1, y1 = int(det.bbox.x1), int(det.bbox.y1)
        x2, y2 = int(det.bbox.x2), int(det.bbox.y2)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = det.category
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        cv2.rectangle(
            img, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1
        )
        cv2.putText(
            img,
            label,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    if save_path:
        save_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, save_img)

    return img


def plot_objects_per_image_histogram(
    annotations: List[ImageAnnotation], save_path: Optional[str] = None
) -> plt.Figure:
    """Plot histogram of number of objects per image.

    Args:
        annotations: List of image annotations.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    objects_per_image = [ann.num_objects for ann in annotations]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(objects_per_image, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    ax.set_title("Distribution of Objects per Image")
    ax.set_xlabel("Number of Objects")
    ax.set_ylabel("Frequency")
    ax.axvline(
        np.mean(objects_per_image),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(objects_per_image):.1f}",
    )
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_occlusion_truncation_stats(
    stats: Dict[str, Dict[str, Any]], save_path: Optional[str] = None
) -> plt.Figure:
    """Plot occlusion and truncation rates per class.

    Args:
        stats: Dictionary with occlusion/truncation statistics.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    classes = list(stats.keys())
    occlusion_rates = [stats[c]["occlusion_rate"] * 100 for c in classes]
    truncation_rates = [stats[c]["truncation_rate"] * 100 for c in classes]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(classes))
    width = 0.35

    ax.bar(x - width / 2, occlusion_rates, width, label="Occlusion Rate", color="coral")
    ax.bar(
        x + width / 2, truncation_rates, width, label="Truncation Rate", color="teal"
    )

    ax.set_xlabel("Class")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Occlusion and Truncation Rates by Class")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
