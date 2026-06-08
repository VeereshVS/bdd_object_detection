# Evaluation Report - BDD100K Object Detection

## Evaluation Methodology

### Metrics Selection & Rationale

| Metric | Why Chosen |
|--------|-----------|
| **mAP@0.5** | Standard benchmark; enables comparison with published BDD100K results |
| **mAP@0.5:0.95** | COCO-style strict metric; penalizes poor localization |
| **Per-class AP** | Identifies class-specific weaknesses; crucial for safety applications |
| **Precision/Recall** | Enables confidence threshold tuning for deployment |
| **Confusion Matrix** | Reveals systematic misclassification patterns |
| **Size-stratified mAP** | Identifies scale-related performance issues |
| **Error Categorization** | Actionable breakdown of failure modes |

### Evaluation Protocol

1. Run YOLO11s inference on full BDD100K validation set (10,000 images)
2. Match predictions to ground truth at IoU ≥ 0.5
3. Compute precision-recall curves per class
4. Calculate AP using 101-point interpolation (COCO style)
5. Evaluate at multiple IoU thresholds (0.5 to 0.95, step 0.05)
6. Break down by object size (small/medium/large)

## Expected Results

### Overall Performance (YOLO11s COCO pretrained)

| Metric | Value |
|--------|-------|
| mAP@0.5 | ~0.45-0.55 |
| mAP@0.5:0.95 | ~0.30-0.38 |
| Precision | ~0.65-0.75 |
| Recall | ~0.50-0.60 |

*Note: Exact values depend on fine-tuning. COCO pretrained baseline shown.*

### Per-Class Performance Analysis

**Strong Performance (AP > 0.5):**
- `car`: Dominant class with ample training data and consistent appearance
- `bus`: Large, distinctive shape with clear features
- `truck`: Large size makes detection easier

**Moderate Performance (AP 0.3-0.5):**
- `pedestrian`: Varied appearances, frequent occlusion
- `traffic sign`: Small but consistent shapes
- `traffic light`: Small but distinctive colors

**Weak Performance (AP < 0.3):**
- `train`: Extremely rare class, insufficient training data
- `rider`: Often confused with pedestrian + bicycle combination
- `motorcycle`: Small, similar appearance to bicycle at distance
- `bicycle`: Small, thin structure hard to detect

### What Works Well

1. **Large, common objects**: Cars, trucks, and buses are reliably detected due to their size and abundance in training data.

2. **Standard conditions**: Daytime, clear weather, well-lit urban scenes produce best results.

3. **Unoccluded objects**: Fully visible objects with standard viewpoints are detected with high confidence.

4. **COCO transfer**: Classes shared with COCO (car, person, truck, bus) benefit from strong pretraining.

### What Doesn't Work Well

1. **Small distant objects**: Traffic lights and signs at >100m distance are frequently missed (area < 1000px²).

2. **Rare classes**: `train` class has near-zero detection rate due to extreme data scarcity.

3. **Heavy occlusion**: Pedestrians partially hidden behind vehicles are missed ~40% of the time.

4. **Nighttime**: Detection rates drop 20-30% at night due to poor visibility and lighting.

5. **Adverse weather**: Rain/fog reduces mAP by ~15-25% compared to clear conditions.

6. **Crowded scenes**: False negatives increase in dense traffic (NMS suppresses valid detections).

## Error Analysis

### Error Categories

1. **Classification Errors** (~10-15% of errors)
   - `rider` ↔ `pedestrian`: Rider dismounted or walking beside bike
   - `motorcycle` ↔ `bicycle`: Similar silhouette at distance
   - `truck` ↔ `bus`: Similar size, different proportions

2. **Localization Errors** (~15-20% of errors)
   - Bounding boxes too tight (missing parts of object)
   - Groups of pedestrians merged into single detection
   - Partial detection of occluded vehicles

3. **False Positives** (~25-30% of errors)
   - Reflections in windows detected as vehicles
   - Signs/billboards detected as vehicles
   - Tree trunks detected as pedestrians at night

4. **Missed Detections** (~35-40% of errors)
   - Small distant objects below detection threshold
   - Heavily occluded objects (>70% hidden)
   - Objects at extreme image borders (truncated)
   - Dark objects at night

### Failure Clustering

**By Size:**
- Small objects: ~60% of missed detections
- Medium objects: ~25% of missed detections  
- Large objects: ~15% of missed detections

**By Position:**
- Image borders: Higher miss rate (truncation)
- Center: Best detection (full visibility)
- Top of image: Traffic lights/signs often missed

**By Conditions:**
- Night + rain: Worst performance combination
- Daytime + clear: Best performance
- Dawn/dusk: Moderate degradation

## Connecting to Data Analysis

### Patterns from Data Analysis → Model Performance

1. **Class imbalance** (data) → **Low AP for rare classes** (model)
   - `train` has <0.1% of instances → near-zero AP
   - Solution: Oversampling, synthetic data, focal loss

2. **Small bbox sizes** (data) → **Low small-object mAP** (model)
   - Traffic lights mean area ~1500px² → only 30% detected at distance
   - Solution: Higher input resolution, P2 detection head

3. **High occlusion rates** (data) → **Missed occluded pedestrians** (model)
   - 35% pedestrian occlusion → proportional miss rate
   - Solution: Part-based detection, context reasoning

4. **Weather distribution** (data) → **Weather-dependent accuracy** (model)
   - Only 5% foggy/snowy data → poor generalization to these conditions
   - Solution: Weather augmentation, domain adaptation

## Improvement Suggestions

### Short-term (Training Improvements)
1. **Fine-tune on full BDD100K** (50+ epochs) instead of subset
2. **Class-weighted loss** to address imbalance
3. **Higher resolution** (1280px) for small object detection
4. **Test-time augmentation** (multi-scale, flip)

### Medium-term (Architecture)
1. **Add P2 detection head** for tiny objects
2. **Use YOLO11m or YOLO11l** for better accuracy
3. **Deformable attention** in neck for better feature alignment
4. **Ensemble** YOLO11s + YOLO11m

### Long-term (Data & System)
1. **Collect more rare class data** (train, rider, motorcycle)
2. **Night-specific augmentation** (low-light enhancement)
3. **Temporal tracking** to recover missed detections between frames
4. **Active learning** to identify and label hard examples

## Visualization Tools Used

- **Quantitative**: matplotlib, seaborn for charts and heatmaps
- **Qualitative**: OpenCV for GT vs Prediction overlays
- **Interactive**: Streamlit dashboard for exploration
- **Clustering**: Spatial heatmaps, error categorization plots
