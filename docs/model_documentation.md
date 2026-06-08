# Model Documentation - YOLO11 for BDD100K Object Detection

## Model Choice: YOLO11s (Small)

### Why YOLO11?

After evaluating several object detection architectures, **YOLO11s** was selected for the following reasons:

| Criteria | YOLO11s | Faster R-CNN | DETR | SSD |
|----------|---------|-------------|------|-----|
| Speed (FPS) | 120+ | 15-20 | 25-30 | 60+ |
| mAP (COCO) | 47.0 | 42.0 | 42.0 | 34.4 |
| Ease of Use | ★★★★★ | ★★★ | ★★★ | ★★★★ |
| Small Objects | ★★★★ | ★★★★★ | ★★★★ | ★★ |
| BDD100K Fit | ★★★★★ | ★★★★ | ★★★ | ★★★ |

**Primary reasons:**

1. **Anchor-free design** eliminates the need for dataset-specific anchor tuning, important given BDD100K's diverse object sizes (tiny traffic lights to large buses).

2. **Multi-scale detection with PANet** provides strong performance across all BDD100K size ranges - P3 head for small objects, P5 for large ones.

3. **COCO pretraining transfer** - 7 of 10 BDD100K classes directly overlap with COCO categories, enabling strong zero-shot transfer.

4. **Practical deployment** - Single-stage detector suitable for real-time driving scenarios.

5. **Active development** - Ultralytics provides excellent tooling, documentation, and community support.

## Architecture Details

### Backbone: CSPDarknet with C3k2 Blocks

```
Input (640×640×3)
    │
    ├── Conv (3×3, stride=2) → 320×320×32
    ├── Conv (3×3, stride=2) → 160×160×64
    ├── C3k2 (n=1) → 160×160×64
    ├── Conv (3×3, stride=2) → 80×80×128
    ├── C3k2 (n=2) → 80×80×128          ← P3 features
    ├── Conv (3×3, stride=2) → 40×40×256
    ├── C3k2 (n=2) → 40×40×256          ← P4 features
    ├── Conv (3×3, stride=2) → 20×20×512
    ├── C3k2 (n=1) → 20×20×512          ← P5 features
    ├── SPPF → 20×20×512
    └── C2PSA → 20×20×512               ← Spatial Attention
```

**C3k2 Module**: Cross-Stage Partial with kernel size 2. An efficient variant that uses two smaller convolutions instead of one large convolution, reducing parameters while maintaining representational capacity.

**C2PSA Module**: Cross Stage Partial with Spatial Attention. Adds spatial attention mechanism at the end of the backbone for better feature focus on relevant regions.

### Neck: PANet with C3k2 Blocks

```
P5 (20×20) ──→ Upsample ──→ Concat with P4 ──→ C3k2 ──→ N4 (40×40)
                                                  │
N4 (40×40) ──→ Upsample ──→ Concat with P3 ──→ C3k2 ──→ N3 (80×80)
                                                  │
N3 (80×80) ──→ Downsample → Concat with N4 ──→ C3k2 ──→ N4' (40×40)
                                                  │
N4'(40×40) ──→ Downsample → Concat with P5 ──→ C3k2 ──→ N5' (20×20)
```

- **Top-down pathway (FPN)**: Upsamples high-level semantic features
- **Bottom-up pathway (PAN)**: Brings spatial detail back to higher levels
- Enables rich multi-scale feature representation

### Head: Decoupled Anchor-Free Head

```
For each scale (P3, P4, P5):
    │
    ├── Classification Branch:
    │   └── Conv → Conv → Sigmoid (num_classes outputs)
    │
    └── Regression Branch:
        ├── Conv → Conv → bbox (4 values: x, y, w, h)
        └── Conv → DFL (Distribution Focal Loss)
```

**Key innovations:**
- **Decoupled head**: Separate branches for classification and localization (better than coupled)
- **Anchor-free**: Directly predicts center offsets and dimensions
- **Distribution Focal Loss (DFL)**: Models bbox boundaries as distributions rather than fixed values

### Loss Functions

1. **CIoU Loss** (Complete IoU): For bounding box regression
   - Considers overlap, distance, and aspect ratio
   - Better convergence than vanilla IoU loss

2. **Binary Cross-Entropy**: For classification
   - Multi-label prediction (object can have multiple attributes)

3. **DFL** (Distribution Focal Loss): For bbox boundary refinement
   - Models each boundary as a discrete probability distribution
   - Better localization for objects with unclear boundaries

### Detection Scales

| Scale | Feature Map | Stride | Best For |
|-------|------------|--------|----------|
| P3 | 80×80 | 8 | Small objects: traffic lights, distant signs |
| P4 | 40×40 | 16 | Medium objects: pedestrians, bicycles |
| P5 | 20×20 | 32 | Large objects: cars, trucks, buses |

## YOLO11s Specifications

| Property | Value |
|----------|-------|
| Parameters | 9.4M |
| GFLOPs | 21.5 |
| Input Size | 640×640 |
| mAP@0.5 (COCO) | 65.0 |
| mAP@0.5:0.95 (COCO) | 47.0 |
| Inference (T4 GPU) | ~2.5ms |

## Training Strategy

### Transfer Learning Approach

1. **Start with COCO pretrained weights** (strong initialization)
2. **Fine-tune all layers** on BDD100K (domain adaptation)
3. **Class mapping**: COCO → BDD100K class correspondence:
   - COCO `person` → BDD100K `pedestrian`
   - COCO `car` → BDD100K `car`
   - COCO `truck` → BDD100K `truck`
   - COCO `bus` → BDD100K `bus`
   - COCO `train` → BDD100K `train`
   - COCO `motorcycle` → BDD100K `motorcycle`
   - COCO `bicycle` → BDD100K `bicycle`
   - COCO `traffic light` → BDD100K `traffic light`

### Training Hyperparameters

```yaml
epochs: 50
batch_size: 16
img_size: 640
optimizer: SGD (momentum=0.937, weight_decay=0.0005)
lr0: 0.01
lrf: 0.01  # final lr = lr0 * lrf
warmup_epochs: 3
augmentation:
  mosaic: 1.0
  mixup: 0.1
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  flipud: 0.0
  fliplr: 0.5
  scale: 0.5
  translate: 0.1
```

### Training Pipeline

```
1. Data Preparation:
   BDD100K JSON → YOLO format (.txt per image)
   Format: class_id x_center y_center width height (normalized)

2. Model Initialization:
   Load YOLO11s with COCO pretrained weights

3. Training:
   - Multi-scale training (random resize 480-800)
   - Mosaic augmentation for context diversity
   - Cosine LR with warmup

4. Validation:
   - Every epoch on BDD100K val set
   - Early stopping (patience=10)
   - Save best model (highest mAP@0.5)
```

## Limitations & Future Work

1. **Input resolution**: 640px may miss very small objects → try 1280px
2. **Class imbalance**: Rare classes (train, rider) need focal loss
3. **Night detection**: Augment with brightness/contrast transforms
4. **Ensemble**: Combine YOLO11s + YOLO11m for better accuracy
5. **Post-processing**: Tracker integration for temporal consistency
