"""YOLO11 Model for BDD100K Object Detection.

This module provides the model setup, architecture explanation, and
inference utilities for YOLO11-based object detection on BDD100K.

Model Choice: YOLO11 (Ultralytics)
===================================

Why YOLO11:
-----------
1. State-of-the-art Performance: YOLO11 achieves improved performance over
   YOLOv8 on COCO with fewer parameters, making it ideal for driving scenes.

2. Architecture Efficiency: Uses C3k2 backbone blocks (efficient small-kernel
   CSP) and C2PSA attention, providing better speed-accuracy trade-off.

3. Anchor-free Design: Like YOLOv8, YOLO11 is anchor-free, reducing
   hyperparameter tuning complexity and improving generalization.

4. Multi-scale Detection: PANet-style feature pyramid with 3 detection
   heads (P3/8, P4/16, P5/32) handles objects at various scales - critical
   for BDD100K where objects range from tiny traffic lights to large trucks.

5. Pre-trained Availability: Models pre-trained on COCO provide strong
   transfer learning for BDD100K's 10 detection classes since most classes
   overlap with COCO categories.

Architecture Overview:
---------------------
- Backbone: CSPDarknet with C3k2 blocks (Cross Stage Partial, kernel size 2)
- Attention: C2PSA (Cross Stage Partial with Spatial Attention) at backbone end
- Neck: PANet with C3k2 blocks for multi-scale feature fusion
- Head: Decoupled head with separate classification and regression branches
- Loss: CIoU loss for bounding box regression, BCE for classification

Model Variants:
- YOLO11n (Nano): 2.6M params, fastest, good for edge deployment
- YOLO11s (Small): 9.4M params, balanced speed/accuracy
- YOLO11m (Medium): 20.1M params, good accuracy
- YOLO11l (Large): 25.3M params, high accuracy
- YOLO11x (Extra-Large): 56.9M params, best accuracy

For BDD100K, we use YOLO11s as the default for its balance of:
- Detection accuracy on small objects (traffic lights, signs)
- Inference speed suitable for driving scenarios
- Reasonable training time on consumer hardware
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

from .dataset import BDD100K_CLASSES, CLASS_TO_ID


class BDD100KDetector:
    """yolo11-based object detector for BDD100K dataset.

    This class wraps the Ultralytics yolo11 model for object detection
    on driving scene images from the BDD100K dataset.

    Attributes:
        model: Ultralytics YOLO model instance.
        model_variant: Model variant name (e.g., 'yolo11s').
        classes: List of BDD100K detection class names.
        conf_threshold: Confidence threshold for detections.
        iou_threshold: IoU threshold for NMS.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_variant: str = "yolo11s",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        """Initialize the BDD100K detector.

        Args:
            model_path: Path to trained weights OR model name (e.g. 'yolo11m').
                If a local file exists, loads it directly. If not, passes to
                Ultralytics for auto-download. If None, uses model_variant.
            model_variant: YOLO11 variant (n/s/m/l/x). Used only when model_path is None.
            conf_threshold: Confidence threshold for predictions.
            iou_threshold: IoU threshold for NMS.
        """
        self.model_variant = model_variant
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = BDD100K_CLASSES

        if model_path:
            # If it's an existing file, load directly; otherwise pass to
            # Ultralytics which handles downloading model names (e.g. "yolo11m")
            if Path(model_path).exists():
                self.model = YOLO(model_path)
            else:
                # Ensure .pt extension for model variant names
                name = model_path if model_path.endswith(".pt") else f"{model_path}.pt"
                self.model = YOLO(name)
        else:
            # Load default pretrained model (COCO weights)
            self.model = YOLO(f"{model_variant}.pt")

    def predict(
        self,
        image_path: str,
        save: bool = False,
        save_dir: Optional[str] = None,
    ) -> Dict:
        """Run inference on a single image.

        Args:
            image_path: Path to the input image.
            save: Whether to save annotated image.
            save_dir: Directory to save results.

        Returns:
            Dictionary with detection results.
        """
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            save=save,
            project=save_dir,
            verbose=False,
        )

        return self._parse_results(results[0])

    def predict_batch(
        self,
        image_paths: List[str],
        batch_size: int = 16,
    ) -> List[Dict]:
        """Run inference on a batch of images.

        Args:
            image_paths: List of image paths.
            batch_size: Batch size for inference.

        Returns:
            List of detection result dictionaries.
        """
        all_results = []

        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i: i + batch_size]
            results = self.model.predict(
                source=batch,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )
            for result in results:
                all_results.append(self._parse_results(result))

        return all_results

    def _parse_results(self, result) -> Dict:
        """Parse yolo11 result object into a structured dictionary.

        Args:
            result: Ultralytics Results object.

        Returns:
            Dictionary with parsed detection information.
        """
        detections = []
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = result.names.get(cls_id, f"class_{cls_id}")

                detections.append(
                    {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": cls_name,
                    }
                )

        return {
            "image_path": result.path,
            "image_shape": result.orig_shape,
            "detections": detections,
            "num_detections": len(detections),
        }

    def train(
        self,
        dataset_yaml: str,
        epochs: int = 1,
        batch_size: int = 16,
        img_size: int = 640,
        project: str = "runs/train",
        name: str = "bdd100k_yolo11",
        device: Optional[str] = None,
        resume: bool = False,
    ) -> str:
        """Train the model on BDD100K dataset.

        Args:
            dataset_yaml: Path to dataset YAML configuration.
            epochs: Number of training epochs.
            batch_size: Training batch size.
            img_size: Input image size.
            project: Project directory for saving results.
            name: Experiment name.
            device: Device to use (cuda/cpu/mps).
            resume: Whether to resume training.

        Returns:
            Path to the best model weights.
        """
        train_args = {
            "data": dataset_yaml,
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": img_size,
            "project": project,
            "name": name,
            "device": device or "cpu",
            "resume": resume,
            "patience": 10,
            "save": True,
            "plots": True,
            "verbose": True,
        }

        results = self.model.train(**train_args)

        # Return path to best weights
        best_weights = Path(project) / name / "weights" / "best.pt"
        return str(best_weights)

    def validate(
        self,
        dataset_yaml: str,
        split: str = "val",
        batch_size: int = 16,
        img_size: int = 640,
        device: Optional[str] = None,
    ) -> Dict:
        """Validate the model on a dataset split.

        Args:
            dataset_yaml: Path to dataset YAML configuration.
            split: Dataset split to validate on.
            batch_size: Batch size for validation.
            img_size: Input image size.
            device: Device to use.

        Returns:
            Dictionary with validation metrics.
        """
        results = self.model.val(
            data=dataset_yaml,
            split=split,
            batch=batch_size,
            imgsz=img_size,
            device=device or "cpu",
            verbose=True,
        )

        metrics = {
            "mAP50": float(results.box.map50),
            "mAP50-95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
            "per_class_ap50": {},
            "per_class_ap50_95": {},
        }

        # Per-class metrics
        if hasattr(results.box, "ap50") and results.box.ap50 is not None:
            for idx, cls_name in enumerate(BDD100K_CLASSES):
                if idx < len(results.box.ap50):
                    metrics["per_class_ap50"][cls_name] = float(
                        results.box.ap50[idx]
                    )
                if idx < len(results.box.ap):
                    metrics["per_class_ap50_95"][cls_name] = float(
                        results.box.ap[idx]
                    )

        return metrics

    def export(self, format: str = "onnx", img_size: int = 640) -> str:
        """Export model to different formats.

        Args:
            format: Export format (onnx, torchscript, tflite, etc.).
            img_size: Input image size for export.

        Returns:
            Path to exported model.
        """
        path = self.model.export(format=format, imgsz=img_size)
        return str(path)


def get_model_info() -> str:
    """Get detailed model architecture information.

    Returns:
        String with model architecture explanation.
    """
    info = """
    YOLO11 Architecture for BDD100K Object Detection
    ==================================================

    1. BACKBONE: CSPDarknet with C3k2 Blocks
       - Input: 640x640x3 (resized from 1280x720)
       - Conv layers with SiLU activation
       - C3k2 blocks (Cross Stage Partial with small kernel size 2)
       - Progressive downsampling: /2, /4, /8, /16, /32
       - C2PSA (Spatial Attention) at backbone end
       - Output feature maps at P3 (80x80), P4 (40x40), P5 (20x20)

    2. NECK: PANet with C3k2 Blocks
       - Top-down pathway (FPN): Upsamples high-level features
       - Bottom-up pathway (PAN): Brings back spatial details
       - Concatenation + C3k2 blocks at each level
       - Enables multi-scale feature fusion

    3. HEAD: Decoupled Detection Head (Anchor-free)
       - Separate branches for classification and regression
       - Classification: Sigmoid output for multi-label prediction
       - Regression: Direct bbox coordinate prediction (x, y, w, h)
       - No anchor boxes (unlike YOLOv5)
       - Distribution Focal Loss for regression

    4. DETECTION SCALES:
       - P3/8 (80x80): Small objects (traffic lights, signs at distance)
       - P4/16 (40x40): Medium objects (pedestrians, bicycles)
       - P5/32 (20x20): Large objects (cars, trucks, buses)

    5. LOSS FUNCTIONS:
       - Box Loss: CIoU (Complete IoU) for bbox regression
       - Classification Loss: Binary Cross-Entropy
       - DFL (Distribution Focal Loss) for bbox distribution

    6. POST-PROCESSING:
       - Non-Maximum Suppression (NMS) with IoU threshold
       - Confidence thresholding

    Why suitable for BDD100K:
    - Multi-scale detection handles diverse object sizes in driving scenes
    - Anchor-free design adapts to BDD100K's varied aspect ratios
    - Fast inference for real-time driving applications
    - Strong transfer from COCO pre-training (overlapping classes)
    """
    return info
