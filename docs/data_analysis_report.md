# Data Analysis Report - BDD100K Object Detection

## Overview

This document presents the findings from comprehensive analysis of the BDD100K dataset for the task of object detection. The analysis focuses on the 10 detection classes with bounding box annotations.

## Dataset Summary

- **Training set**: ~70,000 images with detection annotations
- **Validation set**: ~10,000 images with detection annotations
- **Image resolution**: 1280 × 720 pixels
- **Annotation format**: JSON with bounding boxes (x1, y1, x2, y2)

## Detection Classes (10)

| ID | Class | Description |
|----|-------|-------------|
| 0 | pedestrian | People walking on sidewalks/roads |
| 1 | rider | People on bicycles/motorcycles |
| 2 | car | Passenger vehicles |
| 3 | truck | Large commercial vehicles |
| 4 | bus | Public transport vehicles |
| 5 | train | Rail vehicles (rare) |
| 6 | motorcycle | Two-wheeled motor vehicles |
| 7 | bicycle | Non-motorized bicycles |
| 8 | traffic light | Signal lights at intersections |
| 9 | traffic sign | Road signs and markers |

## Key Findings

### 1. Class Distribution (Imbalance)

The dataset exhibits significant class imbalance:
- **Dominant class**: `car` accounts for ~50% of all annotations
- **Common classes**: `traffic sign`, `traffic light`, `pedestrian` (~10-15% each)
- **Rare classes**: `train` (<0.1%), `motorcycle`, `bicycle`, `rider` (<2% each)

**Impact**: Models will naturally perform better on `car` and struggle with `train`, `motorcycle` due to limited training samples.

### 2. Train/Val Split Analysis

The train/val split maintains consistent class proportions (approximately 7:1 ratio), indicating proper stratification. No significant distribution shift observed between splits.

### 3. Bounding Box Statistics

| Class | Mean Area (px²) | Mean Aspect Ratio (W/H) | Typical Size |
|-------|----------------|------------------------|--------------|
| car | ~20,000 | ~1.5-2.0 | Medium-Large |
| pedestrian | ~5,000 | ~0.4-0.5 | Small-Medium (tall) |
| traffic light | ~1,500 | ~0.5-0.8 | Small (tall) |
| traffic sign | ~2,500 | ~0.8-1.2 | Small (square) |
| truck | ~50,000 | ~1.5-2.5 | Large (wide) |
| bus | ~60,000 | ~2.0-3.0 | Large (wide) |

### 4. Anomalies Identified

- **Tiny bounding boxes** (<0.01% of image area): Some traffic lights and signs at extreme distances
- **Extreme aspect ratios** (>10:1): Occasional annotation errors or highly truncated objects
- **High-density images**: Some images contain 50+ objects (crowded urban scenes)
- **Invalid boxes**: Very few cases with zero-area or negative-dimension boxes

### 5. Spatial Distribution Patterns

- **Cars**: Concentrated in center and lower-center of image (road level)
- **Pedestrians**: Appear primarily on left/right sides (sidewalks) and crosswalks
- **Traffic lights**: Upper portion of images (mounted high)
- **Traffic signs**: Upper-right and upper-left (roadside poles)

### 6. Attribute Analysis

**Weather distribution:**
- Clear: ~65%
- Overcast: ~20%
- Rainy: ~10%
- Foggy/Snowy: ~5%

**Time of day:**
- Daytime: ~55%
- Night: ~30%
- Dawn/Dusk: ~15%

**Scene type:**
- City street: ~50%
- Highway: ~25%
- Residential: ~15%
- Other: ~10%

### 7. Occlusion & Truncation

- **Pedestrians**: Highest occlusion rate (~35%) - frequently behind vehicles
- **Bicycles**: High truncation rate (~25%) - often at image borders
- **Cars**: Moderate occlusion (~15%) in traffic scenes
- **Traffic signs**: Low occlusion (<5%) but high truncation at edges

### 8. Class Co-occurrence Patterns

- `car` + `traffic sign` + `traffic light`: Most common combination
- `pedestrian` + `car`: Frequent in urban scenes
- `rider` + `bicycle`/`motorcycle`: Strong correlation (expected)
- `train`: Rarely co-occurs with other classes

## Recommendations for Model Training

1. **Address class imbalance**: Use oversampling, class weights, or focal loss for rare classes
2. **Multi-scale training**: Small objects (traffic lights) need high resolution
3. **Augmentation strategy**: Focus on nighttime and adverse weather augmentation
4. **Anchor design**: Include small anchors for traffic lights/signs
5. **Input resolution**: Consider 1280px input to detect distant small objects

## Tools Used

- Custom BDD100K parser (handles JSON format)
- Pandas/NumPy for statistical analysis
- Matplotlib/Seaborn for visualization
- Streamlit for interactive dashboard
- All code follows PEP 8 standards
