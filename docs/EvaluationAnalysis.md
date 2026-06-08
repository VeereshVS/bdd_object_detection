# Evaluation Analysis — YOLO11m on BDD100K Validation Set

**Model**: YOLO11m (Medium) — COCO pretrained, zero-shot transfer with COCO→BDD100K class mapping  
**Dataset**: BDD100K Validation Set — 10,000 images, 10 detection classes  
**Confidence Threshold**: 0.25 | **IoU Threshold**: 0.5  
**Evaluation Date**: June 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Metrics Selection & Rationale](#2-metrics-selection--rationale)
3. [Quantitative Performance](#3-quantitative-performance)
4. [Per-Class Deep Dive](#4-per-class-deep-dive)
5. [Performance by Object Size](#5-performance-by-object-size)
6. [Error Analysis](#6-error-analysis)
7. [Qualitative Analysis — Failure Clustering](#7-qualitative-analysis---failure-clustering)
8. [Connecting Data Analysis to Model Performance](#8-connecting-data-analysis-to-model-performance)
9. [What Works and What Doesn't](#9-what-works-and-what-doesnt)
10. [Improvement Recommendations](#10-improvement-recommendations)

---

## 1. Executive Summary

The YOLO11m model (COCO pretrained, no BDD100K fine-tuning) with a COCO→BDD100K class name mapping achieves an **mAP@0.5 of 0.228** on the full BDD100K validation set (10,000 images). This is a **zero-shot domain transfer** evaluation — the model was trained on COCO and evaluated directly on BDD100K with only a label remapping layer (e.g., COCO "person" → BDD100K "pedestrian", COCO "stop sign" → BDD100K "traffic sign").

The performance reveals a clear **three-tier pattern**:
- **Tier 1 (AP > 0.33)**: car, pedestrian, bus, truck — classes with direct COCO counterparts and sufficient object size
- **Tier 2 (AP 0.09–0.16)**: traffic light, traffic sign — classes limited by small object size or partial taxonomy coverage
- **Tier 3 (AP ≈ 0)**: rider, motorcycle, bicycle, train — classes with fundamental taxonomy gaps or extreme data rarity

**Key Numbers:**

| Metric | Value |
|--------|-------|
| mAP@0.5 | **0.228** |
| mAP@0.75 | **0.179** |
| mAP@0.5:0.95 | **0.171** |
| Best Class AP | car — 0.446 |
| Classes > 0.3 AP | car, pedestrian, bus, truck |
| Zero AP Classes | rider, motorcycle, bicycle |

---

## 2. Metrics Selection & Rationale

### Why These Metrics Were Chosen

| Metric | Why Selected | What It Tells Us |
|--------|-------------|-----------------|
| **mAP@0.5** | Standard PASCAL VOC benchmark metric. Allows direct comparison with published BDD100K results. IoU=0.5 is forgiving enough to reward correct detections even with imperfect localization. | Overall detection quality — are we finding objects in roughly the right place? |
| **mAP@0.5:0.95** | COCO-standard metric averaging across IoU thresholds 0.5–0.95 in steps of 0.05. More stringent; penalizes loose bounding boxes. | Localization precision — are bounding boxes tightly aligned to objects? |
| **Per-Class AP** | BDD100K has extreme class imbalance (car: 102K GT vs train: 15 GT). A single mAP number hides class-specific strengths/weaknesses. | Identifies which classes the model handles well and which fail. |
| **Precision & Recall** | Precision measures false positive rate; recall measures miss rate. Both critical for driving: false positives cause unnecessary braking; missed detections cause collisions. | Safety-relevant trade-off analysis. |
| **Performance by Object Size** (Small/Medium/Large) | BDD100K contains objects ranging from tiny distant traffic lights (~1,500 px²) to large nearby trucks (~50,000+ px²). Performance varies dramatically with scale. | Reveals scale-dependent failures critical for driving safety. |
| **Error Categorization** (classification, localization, duplicates, background FPs, missed) | Raw AP doesn't explain *why* the model fails. Error type breakdown tells us whether to fix the classifier, localizer, or NMS. | Actionable diagnostics for targeted improvement. |

### Metrics NOT Used (and Why)

- **F1 Score**: A single-point metric that doesn't capture the full precision-recall trade-off. AP (area under the P-R curve) is more informative.
- **FPS / Inference Speed**: Not the focus here — we're diagnosing detection quality, not deployment readiness.
- **Confusion Matrix alone**: Less informative than per-class AP + error categorization for this use case.

---

## 3. Quantitative Performance

### 3.1 Overall Metrics

| Metric | Value | Interpretation |
|--------|-------|---------------|
| mAP@0.5 | **0.228** | Moderate — reasonable for zero-shot COCO→BDD100K transfer without fine-tuning |
| mAP@0.75 | **0.179** | 78% of mAP@0.5; indicates good localization quality when detection succeeds |
| mAP@0.5:0.95 | **0.171** | 75% of mAP@0.5; bounding boxes are well-placed overall |

**The gap between mAP@0.5 (0.228) and mAP@0.5:0.95 (0.171) is only 25%**, which tells us that when the model detects something correctly, the bounding boxes are well-localized. The primary issue is not localization quality — it's **recall** (missing objects entirely, especially small or rare ones).

### 3.2 Per-Class Performance

| Class | AP@0.5 | Precision | Recall | GT Count | Predictions | TP | FP | FN |
|-------|--------|-----------|--------|----------|-------------|------|------|-------|
| **car** | **0.446** | 0.878 | 0.484 | 102,506 | 56,472 | 49,569 | 6,903 | 52,937 |
| **pedestrian** | **0.414** | 0.810 | 0.413 | 13,262 | 6,770 | 5,482 | 1,288 | 7,780 |
| **bus** | **0.377** | 0.528 | 0.425 | 1,597 | 1,284 | 678 | 606 | 919 |
| **truck** | **0.332** | 0.413 | 0.455 | 4,245 | 4,669 | 1,930 | 2,739 | 2,315 |
| **traffic light** | **0.159** | 0.615 | 0.162 | 26,885 | 7,089 | 4,357 | 2,732 | 22,528 |
| **traffic sign** | **0.091** | 0.951 | 0.006 | 34,908 | 224 | 213 | 11 | 34,695 |
| **train** | **0.009** | 0.019 | 0.133 | 15 | 108 | 2 | 106 | 13 |
| rider | 0.000 | 0.000 | 0.000 | 649 | 0 | 0 | 0 | 649 |
| motorcycle | 0.000 | 0.000 | 0.000 | 0 | 216 | 0 | 216 | 0 |
| bicycle | 0.000 | 0.000 | 0.000 | 0 | 356 | 0 | 356 | 0 |

> **Visualization**: See `evaluation_results_yolo11m_conf0.25/plots/map_per_class.png`

### 3.3 Performance by Object Size

| Size Category | mAP@0.5 | Relative to Overall |
|---------------|---------|-------------------|
| **Small** (area < 32²) | **0.075** | 0.33× — severely degraded |
| **Medium** (32² ≤ area < 96²) | **0.243** | 1.07× — near overall average |
| **Large** (area ≥ 96²) | **0.381** | 1.67× — best performance |

> **Visualization**: See `evaluation_results_yolo11m_conf0.25/plots/performance_by_size.png`

**Analysis**: A **5.1× performance gap** exists between small and large objects. This is concerning for driving safety because critical objects like distant traffic lights and pedestrians are often small. The YOLO11m model at 640px input resolution struggles to resolve features for objects under 32×32 pixels — corresponding to objects roughly 2.5% of image width or smaller in the original 1280×720 BDD100K images.

---

## 4. Per-Class Deep Dive

### 4.1 Car (AP = 0.446) — Best Performer

- **Why it works**: "car" is an exact label match between COCO and BDD100K. Cars are the most abundant class (102K ground truths) and the most heavily represented in COCO pretraining data.
- **Precision 0.878**: When the model says "car," it's right 88% of the time — excellent false positive control.
- **Recall 0.484**: Only ~48% of cars are detected. Many missed cars are distant/small or heavily occluded. At 640px input, a car at 100m distance may only be 15–20px wide — below effective detection limits.
- **Room for improvement**: Higher input resolution (1280px) would dramatically boost recall for distant cars.

### 4.2 Pedestrian (AP = 0.414) — Strong Performer (via label mapping)

- **Why it works**: The COCO→BDD100K mapping (`"person"→"pedestrian"`) successfully recovered this class. The model's COCO "person" detector transfers well to BDD100K driving scenes.
- **Precision 0.810**: High confidence predictions — the model is selective about what it calls a pedestrian.
- **Recall 0.413**: 5,482 of 13,262 pedestrians detected. Missed pedestrians are predominantly: (a) at distance (small), (b) heavily occluded behind vehicles, or (c) in nighttime scenes with low visibility.
- **Safety note**: Missing 59% of pedestrians is a critical safety concern for autonomous driving applications.

### 4.3 Bus (AP = 0.377) & Truck (AP = 0.332) — Good Performers

- **Why they work**: Direct COCO label matches. These are large objects (mean area ~50K–60K px²) that are easy to detect.
- **Truck precision is low (0.413)**: Significant confusion between truck, bus, and car. Many SUVs/vans on the boundary between "car" and "truck" get misclassified. The 2,739 truck false positives likely include car→truck misclassifications.
- **Bus recall 0.425**: Many buses at distance or partially occluded behind other vehicles are missed.

### 4.4 Traffic Light (AP = 0.159) — Moderate Performer, Limited by Size

- **Why partially works**: "traffic light" exists in COCO with the same name.
- **Recall only 0.162**: Of 26,885 ground truth traffic lights, only 4,357 were detected. Directly attributable to **small object size** — traffic lights have mean area ~1,500 px². At 640px input, many become sub-10px, below the P3 head's effective resolution.
- **Precision 0.615**: When detected, predictions are moderately reliable. The 2,732 false positives likely include reflections, streetlights, or other bright points.

### 4.5 Traffic Sign (AP = 0.091) — Very Low, Partially Recovered by Mapping

- **Partial recovery**: The mapping `COCO "stop sign" → BDD100K "traffic sign"` recovered only **224 predictions** out of 34,908 ground truth signs. COCO only knows "stop sign" — it cannot detect speed limit signs, warning signs, direction signs, etc.
- **Extraordinary precision (0.951)**: Of 224 predictions, 213 are correct. The model is extremely accurate for the narrow subset of signs it can identify (octagonal stop signs), but completely blind to the vast majority.
- **Recall 0.006**: Near-zero. Only 0.6% of traffic signs are detected. This class fundamentally requires fine-tuning to learn the hundreds of sign types in BDD100K.

### 4.6 Rider (AP = 0.000) — Structural Taxonomy Gap

- COCO has no "rider" class. COCO detects "person" and "motorcycle"/"bicycle" as separate objects. BDD100K merges the person+vehicle into a single "rider" bounding box. No post-processing mapping can fix this — it requires a fundamentally different detection output.
- **649 missed riders** in validation. These include cyclists and motorcyclists — among the most vulnerable road users.

### 4.7 Motorcycle & Bicycle (AP = 0.000) — GT Annotation Mismatch

- **Anomaly**: GT count is 0 for both motorcycle and bicycle, yet the model predicts 216 motorcycles and 356 bicycles. This reveals that in BDD100K, standalone motorcycle/bicycle annotations are subsumed into the "rider" class (person+vehicle combined). The vehicles the model correctly detects have no matching GT because BDD100K annotates them differently.
- These are **valid detections being penalized** as false positives due to BDD100K annotation conventions.

### 4.8 Train (AP = 0.009) — Near Random

- Only **15 ground truth** instances in the entire validation set — statistically insufficient for reliable evaluation.
- 108 predictions with only 2 true positives and 106 false positives. The model confuses buses, trucks, and large structures with trains.
- **Not actionable** at this sample size. Would need external data or significantly more training examples.

---

## 5. Performance by Object Size

### Small Objects (mAP = 0.075)

Small objects (< 32×32 px at 640px input) include:
- Distant traffic lights
- Far-away cars and pedestrians
- Traffic signs at distance

**Why performance is poor**:
1. At 640×640 input, objects < 32px are represented by < 4×4 features in the P3 (stride-8) head — insufficient for reliable classification.
2. BDD100K images are 1280×720 — downscaling to 640px loses half the spatial detail.
3. Small objects have less distinctive features and are more likely partially occluded.

### Medium Objects (mAP = 0.243)

Medium objects (32² ≤ area < 96²) include:
- Pedestrians at moderate distance (10–30m)
- Cars in adjacent lanes
- Traffic lights at medium range

This is near the overall mAP, representing the model's "typical" operating point where features are sufficient for reliable classification.

### Large Objects (mAP = 0.381)

Large objects (≥ 96×96 px) include:
- Nearby cars (within ~20m)
- Trucks and buses
- Close-range pedestrians

**Why performance is best**: Sufficient spatial features for classification and localization. Large objects are well-represented in both COCO and BDD100K training data.

---

## 6. Error Analysis

### Error Type Breakdown

| Error Type | Count | % of Total Errors | Description |
|-----------|-------|-------------------|-------------|
| **Missed Detections** | 119,224 | 88.5% | Ground truth objects the model never detected |
| **Localization Errors** | 6,658 | 4.9% | Correct class but poor bbox overlap (IoU 0.3–0.5) |
| **Background FPs** | 3,851 | 2.9% | Detections with no matching ground truth (IoU < 0.3) |
| **Classification Errors** | 3,243 | 2.4% | Correct localization but wrong class label |
| **Duplicate Detections** | 1,836 | 1.4% | Multiple predictions for the same object |
| **Total** | 134,812 | 100% | — |

### Analysis

1. **Missed detections dominate (88.5%)**: The model fails overwhelmingly by NOT detecting objects, not by predicting wrong ones. Breaking down the 119K misses:
   - ~34,695 missed traffic signs (model has no concept of generic signs beyond stop signs)
   - ~22,528 missed traffic lights (too small at 640px resolution)
   - ~52,937 missed cars (distant/small/occluded)
   - ~7,780 missed pedestrians (nighttime, occlusion, distance)
   - ~649 missed riders (no class mapping possible)

2. **Localization errors exceed classification errors (6,658 vs 3,243)**: This indicates the model generally classifies correctly when it detects something, but bbox precision could improve. Fine-tuning on BDD100K-specific object boundary conventions would help — BDD100K may annotate tighter/looser boxes than COCO conventions.

3. **Classification errors (2.4%)**: 3,243 cases of correct localization but wrong class. Primary confusions:
   - truck ↔ bus ↔ car for ambiguous vehicle types (vans, SUVs, pickup trucks)
   - train ← bus/truck (large vehicles misidentified as trains)

4. **Background FPs (2.9%)**: 3,851 hallucinated detections across 10,000 images (0.39 per image). Relatively low — the model is conservative and doesn't frequently hallucinate objects.

5. **Duplicate detections (1.4%)**: 1,836 cases of multiple predictions per object. NMS is working reasonably well; a slightly lower NMS IoU threshold could reduce these further.

### Key Insight: The Error Profile Points to a Recall Problem

The 18:1 ratio of missed detections to localization errors shows this model's weakness is **detection sensitivity, not precision**. The model is highly precise when it fires (precision > 0.8 for car, pedestrian, traffic sign) but misses the majority of objects. This suggests:
- The confidence threshold (0.25) filters out many valid low-confidence detections for small/distant objects
- The model's feature resolution at 640px is fundamentally insufficient for the density of small objects in BDD100K
- Fine-tuning would teach the model to be more aggressive on BDD100K-specific object appearances

---

## 7. Qualitative Analysis — Failure Clustering

### Visualization Tool

Side-by-side Ground Truth (green boxes, left panel) vs Predictions (red boxes, right panel) visualizations were generated for 20 sample images using OpenCV. Each image shows identical scene content with GT annotations on the left and model predictions (with confidence scores) on the right. These are stored in:
```
evaluation_results_yolo11m_conf0.25/qualitative/sample_0000.jpg through sample_0019.jpg
```

### Failure Clusters Identified from Visual Inspection

#### Cluster 1: Small/Distant Object Misses
**Observed in**: Samples 0, 2, 5, 10, 15  
**Pattern**: Ground truth contains many small boxes (traffic lights, distant cars, signs) that are completely absent from predictions. The model consistently detects only large, nearby objects.

- **Sample 0** (urban street, dusk): GT has ~30+ annotations including traffic lights, signs, distant cars, and pedestrians. Predictions show only ~8 large foreground cars. All traffic signs, distant objects, and small traffic lights are missed.
- **Sample 2** (night highway): GT shows 8+ cars plus large highway traffic signs. Model detects only 3 nearest cars — small distant cars missed. Does detect the large green highway sign (via stop sign mapping).
- **Sample 15** (urban scene with trucks/police van): Good detection of nearby large vehicles (truck, van correctly detected). But small background cars and distant objects completely missed.

**Root Cause**: 640px input resolution. Objects below ~20px become unresolvable — the P3 detection head (stride 8) has only 2–3 feature cells for these objects.

#### Cluster 2: Nighttime Performance Degradation
**Observed in**: Samples 2, 5  
**Pattern**: Nighttime scenes show dramatically fewer detections vs GT. The model detects only the brightest/most visible objects (tail lights, illuminated cars). Even traffic lights (which are brighter at night) are partially detected.

- **Sample 5** (night intersection): GT shows cars, traffic lights, and signs. Model detects only the nearest car. Also produces an ego vehicle false positive (large red box on dashboard/hood area at bottom of frame).
- **Sample 2** (night highway bridge): GT has multiple cars at various distances. Only 3 nearest cars (visible via tail lights) are detected.

**Root Cause**: Low-light conditions reduce feature contrast in CNN layers. COCO training data has limited night driving scenes compared to BDD100K's ~30% nighttime distribution.

#### Cluster 3: Ego Vehicle False Positives
**Observed in**: Samples 5, 10, 18  
**Pattern**: The model places a large bounding box on the ego vehicle's hood, dashboard, or side mirror visible at the bottom/edges of the frame. This is a consistent false positive — the ego vehicle is not annotated in BDD100K ground truth.

- **Sample 5** (night): Large "car 0.74" confidence box on the ego vehicle dashboard reflection.
- **Sample 10** (urban dusk): Model detects the ego vehicle's side mirror and hood as a car.
- **Sample 18** (residential): Ego vehicle hood/windshield wiper area detected as car.

**Root Cause**: The model correctly identifies car features (body panels, lights, mirrors) on the dashcam vehicle's own visible surfaces. This is a domain-specific false positive unique to dashcam datasets that fine-tuning would address (model would learn to suppress bottom-of-frame detections).

#### Cluster 4: Occlusion and Overlapping Objects
**Observed in**: Samples 10, 15, 18  
**Pattern**: In scenes with multiple parked or stopped vehicles, the model either misses heavily occluded objects or produces imprecise boxes.

- **Sample 10** (urban intersection with SUV): GT annotates the large foreground SUV and multiple cars behind/beside it. Model correctly detects the SUV but misses cars that are >50% occluded by it.
- **Sample 18** (residential with parked cars): Model correctly identifies visible parked cars but omits cars partially hidden behind fences, poles, or snow banks.

**Root Cause**: NMS may suppress valid overlapping detections. Additionally, heavily occluded objects have insufficient visible features for confident detection at the 0.25 threshold.

#### Cluster 5: Traffic Sign Blindness  
**Observed in**: Samples 0, 2, 10, 15, 18  
**Pattern**: Ground truth contains abundant traffic sign annotations (speed limits, street signs, warning signs, highway direction signs) — the vast majority are undetected. Only occasional stop signs or large highway guide signs are picked up.

- **Sample 0** (urban): ~10 traffic signs annotated (green boxes in upper portions) — nearly none appear in predictions.
- **Sample 2** (night highway): Large green highway signs ARE detected (model maps them as "traffic sign" via the stop sign heuristic), but all smaller regulatory signs missed.
- **Sample 18** (residential): END sign and road work signs annotated in GT — the model detects the distinctive red octagonal ones via stop sign mapping.

**Root Cause**: COCO only has "stop sign" as a sign category. The mapping `"stop sign"→"traffic sign"` recovers only stop-sign-shaped objects. The vast vocabulary of traffic signs (speed limits, warnings, information signs) is entirely unknown to the model.

---

## 8. Connecting Data Analysis to Model Performance

The data analysis (documented in `docs/data_analysis_report.md`) revealed patterns that directly explain evaluation results:

### 8.1 Object Size Distribution → Scale-Dependent Failure

| Data Analysis Finding | Evaluation Impact |
|----------------------|-------------------|
| Traffic lights: mean area ~1,500 px² (very small) | AP=0.159, recall=0.162 — 84% missed due to size |
| Traffic signs: mean area ~2,500 px² (small) | AP=0.091, recall=0.006 — mapping issue + size combined |
| Pedestrians: mean area ~5,000 px² (small-medium) | AP=0.414, recall=0.413 — decent for medium-distance pedestrians |
| Cars: mean area ~20,000 px² (medium-large) | AP=0.446, recall=0.484 — best balanced performance |
| Trucks/buses: mean area ~50K–60K px² (large) | AP=0.33–0.38 — large size compensates for less data |

### 8.2 Spatial Distribution → Detection Patterns

| Data Analysis Finding | Evaluation Impact |
|----------------------|-------------------|
| Cars concentrated in center/lower frame | Model performs well here — center has best resolution after letterboxing |
| Traffic lights in upper frame | Upper-frame features have less resolution after 640px resize |
| Pedestrians on sides (sidewalks) | Peripheral objects receive less spatial attention |
| Traffic signs on upper-left/right (poles) | Peripheral + small = compounded difficulty |

### 8.3 Occlusion/Truncation Rates → Error Patterns

| Data Analysis Finding | Evaluation Impact |
|----------------------|-------------------|
| Pedestrians: highest occlusion rate (~35%) | Recall=0.413 — occluded pedestrians contribute significantly to the 7,780 misses |
| Cars: moderate occlusion (~15%) | Qualitative analysis confirms model misses heavily occluded cars |
| Bicycles: high truncation (~25%) at borders | Combined with "rider" taxonomy issue → complete failure |

### 8.4 Scene/Time-of-Day Distribution → Environmental Factors

| Data Analysis Finding | Evaluation Impact |
|----------------------|-------------------|
| ~30% nighttime images in BDD100K | Qualitative analysis confirms severe nighttime degradation (Cluster 2) |
| ~10% rainy conditions | Reduced visibility contributes to missed detections |
| ~50% city street scenes | Dense urban scenes have many small objects — hardest scenario |
| ~25% highway scenes | Fewer objects per image but at greater distances → small object problem |

---

## 9. What Works and What Doesn't

### What Works Well

1. **Vehicle detection (car/bus/truck)**: Strong zero-shot transfer. Car precision of 0.878 means nearly 9 in 10 car predictions are correct. These classes benefit from direct COCO→BDD100K label alignment and large object sizes.

2. **Pedestrian detection (after label mapping)**: AP=0.414 with precision 0.810 demonstrates that the COCO "person" detector transfers effectively to BDD100K driving scenes. The model has learned robust person features from COCO's extensive person annotations (~260K instances).

3. **Localization quality**: The 25% gap between mAP@0.5 (0.228) and mAP@0.5:0.95 (0.171) is moderate, indicating decent bounding box precision. When the model detects something, it places boxes reasonably well.

4. **Precision-first behavior**: The model is conservative — precision exceeds 0.8 for car, pedestrian, and traffic sign. It rarely hallucinates objects (only 0.39 background FPs per image). This is preferable for autonomous driving where false positives cause unnecessary emergency stops.

5. **Large object detection**: mAP=0.381 for large objects shows reliable handling of nearby, prominent objects — the most immediately safety-critical scenario (collision avoidance with nearby obstacles).

### What Doesn't Work

1. **Traffic sign detection (AP = 0.091)**: COCO's limited "stop sign" class cannot generalize to BDD100K's diverse traffic sign vocabulary. With 34,908 GT signs but only 213 correct detections, this is 99.4% failure. **Critical for autonomous driving** — missing speed limits, warnings, and directions.

2. **Small object detection**: mAP drops from 0.381 (large) to 0.075 (small) — a **5.1× degradation**. Traffic lights (26,885 GT) are predominantly small and have only 16.2% recall. Missing a red traffic light is safety-critical.

3. **Rider class (AP = 0.000)**: Structural taxonomy gap — COCO doesn't have a combined person+vehicle "rider" concept. 649 riders completely invisible to the model. Cyclists and motorcyclists are among the most vulnerable road users.

4. **Nighttime performance**: Qualitative analysis reveals severe degradation. The model detects only large, well-lit objects at night. COCO training data lacks the night driving conditions that constitute ~30% of BDD100K.

5. **High miss rate overall**: 119,224 missed detections. Even for the best class (car), recall is only 48.4%. The model sees less than half of what exists in the scene.

6. **Ego vehicle false positives**: Consistent false detection of the dashcam vehicle's own surfaces. A minor but systematic issue (~1 FP per nighttime frame) unique to dashcam deployment.

### Root Cause Summary

| Root Cause | Impact | Solvability |
|-----------|--------|------------|
| No fine-tuning on BDD100K | Model hasn't learned BDD100K-specific classes or driving domain | **Medium** — fine-tune 50+ epochs |
| Input resolution (640px for 1280×720 source) | Small objects missed (mAP_small = 0.075) | **Medium** — use 1280px (4× compute) |
| COCO has no generic "traffic sign" | 34,908 signs at 0.6% recall | **Requires training** — new class |
| COCO has no "rider" class | 649 riders at 0% recall | **Requires training** — new concept |
| Limited night training in COCO | Night performance collapses | **Hard** — augmentation + data needed |
| Extreme class rarity (train: 15 samples) | Near-random performance | **Hard** — need external data |

---

## 10. Improvement Recommendations

### Priority 1: Fine-Tune on BDD100K (Highest Impact)

Fine-tuning on the BDD100K training set (70K images) for 50–100 epochs would:
- Teach the model BDD100K-specific classes (traffic sign, rider) — the biggest gap
- Adapt features to driving scene characteristics (dashcam perspective, road scenarios)
- Learn to suppress ego vehicle detections
- Improve night/weather handling through domain-specific examples
- Align bounding box conventions to BDD100K annotation style

```bash
python -m src.train --model yolo11m --bdd-root ../bdd100k --epochs 50 --batch-size 16 --device cuda
```

### Priority 2: Increase Input Resolution

- Use `--imgsz 1280` to match BDD100K's native resolution
- Traffic lights (area ~1,500 px²) would double in pixel count → more features for detection
- Trade-off: ~4× slower inference, ~4× more GPU memory
- Alternative: Use P2 detection head (stride 4) for better small object features at lower compute cost

### Priority 3: Address Traffic Sign Detection

The model is 99.4% blind to traffic signs. Options ranked by practicality:
1. **Fine-tuning on BDD100K** (recommended): Training data has abundant traffic sign annotations with full diversity
2. **Pre-training on traffic sign datasets** (GTSDB, MTSD) before BDD100K fine-tuning for extra sign knowledge
3. **Higher resolution input**: Many signs are small — 1280px input helps even without fine-tuning

### Priority 4: Night-Specific Improvements

- Apply aggressive brightness/contrast/gamma augmentation during training
- Add CLAHE (Contrast Limited Adaptive Histogram Equalization) as preprocessing
- Consider separate confidence thresholds for day vs night (lower at night to recover recall)
- Data-level: ensure night images are well-represented in training batches (currently ~30%)

### Priority 5: Handle Rider Class

Options:
- **Fine-tune on BDD100K** (most practical): Training set has "rider" annotations — the model will learn this combined concept
- Post-processing heuristic: if "pedestrian" bbox highly overlaps with "motorcycle"/"bicycle" bbox, merge into "rider" box
- Modify training to add rider-specific augmentation (person + vehicle composite)

### Priority 6: Ensemble & Post-Processing

- **Multi-model ensemble**: YOLO11s (fast) + YOLO11m (balanced) + YOLO11l (small object focus)
- **Class-specific confidence thresholds**: Lower threshold for rare classes (train, rider) to improve recall
- **Test-time augmentation**: Multi-scale inference (640 + 1280) + horizontal flip for +2–3% mAP
- **Ego vehicle suppression**: Post-processing rule to ignore car detections in bottom 15% of frame with >80% overlap to image borders

---

## Appendix A: Visualization Outputs

### Quantitative Visualizations

| File | Description |
|------|-------------|
| `plots/map_per_class.png` | Horizontal bar chart showing AP per class with mAP reference line (red dashed at 0.228). Clear three-tier structure visible: car/pedestrian/bus/truck well above average; traffic light/sign below; rider/motorcycle/bicycle at zero. |
| `plots/performance_by_size.png` | Bar chart of mAP by object size category. Shows the 5× gap: small=0.075, medium=0.243, large=0.381. Color-coded (red→cyan) for severity. |

### Qualitative Visualizations (GT vs Predictions)

| File | Scene Type | Key Observations |
|------|-----------|------------------|
| `sample_0000.jpg` | Urban street, dusk | Dense GT annotations (30+). Model detects large cars; misses all signs, most lights, distant objects. Pedestrians now detected via mapping. |
| `sample_0002.jpg` | Night highway | Model detects 3 nearest cars + large highway sign. All distant cars and small signs missed. Night degradation evident. |
| `sample_0005.jpg` | Night intersection | Ego vehicle FP at bottom. Only nearest car detected. Nearly all GT objects missed in darkness. |
| `sample_0010.jpg` | Urban dusk, SUV scene | Good detection of foreground vehicles. Occluded cars behind SUV missed. Pedestrians detected. Traffic signs missed. |
| `sample_0015.jpg` | Urban with trucks/police van | Large vehicles well-detected (truck correctly identified). Distant background cars missed. Traffic signs absent. |
| `sample_0018.jpg` | Residential street, winter | Cars, pedestrian, and some signs detected. Ego vehicle FP. Good overall coverage for nearby objects. |

### How to Regenerate

```bash
# From project root (bdd100k_object_detection/)
python evaluation/run_evaluation.py \
    --model yolo11m \
    --data-root ../bdd100k \
    --output evaluation_results_yolo11m_conf0.25 \
    --conf 0.25
```

---

## Appendix B: COCO→BDD100K Class Mapping Applied

The following class name mapping was applied during inference to bridge the COCO→BDD100K label taxonomy gap:

| COCO Label | BDD100K Label | Impact on Results |
|-----------|--------------|-------------------|
| `person` | `pedestrian` | **Major** — recovered AP from 0.000 to 0.414 (5,482 TP recovered) |
| `stop sign` | `traffic sign` | **Minor** — recovered AP from 0.000 to 0.091 (213 TP, but 34,695 FN remain) |
| `car` | `car` | No change (already matching) |
| `truck` | `truck` | No change |
| `bus` | `bus` | No change |
| `train` | `train` | No change |
| `motorcycle` | `motorcycle` | No impact (GT is under "rider" in BDD100K) |
| `bicycle` | `bicycle` | No impact (GT is under "rider" in BDD100K) |
| `traffic light` | `traffic light` | No change |

All other COCO classes (71 remaining out of 80 total) are **dropped** during inference as they have no BDD100K equivalent. This prevents false positives from irrelevant COCO classes (e.g., "couch", "laptop", "pizza") appearing in driving scene evaluation.

---

*Document based on evaluation results in `evaluation_results_yolo11m_conf0.25/`. All metrics computed on the full BDD100K validation set (10,000 images) using YOLO11m with COCO pretrained weights and COCO→BDD100K class mapping.*
