# BDD100K Object Detection - End-to-End Pipeline

A comprehensive object detection project on the [BDD100K](https://www.bdd100k.com/) dataset, covering data analysis, model training with YOLO11, and evaluation with visualization.

## Table of Contents
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Task 1: Data Analysis](#task-1-data-analysis)
- [Task 2: Model](#task-2-model)
- [Task 3: Evaluation & Visualization](#task-3-evaluation--visualization)
- [Docker Usage](#docker-usage)
- [Google Colab](#google-colab)
- [Requirements](#requirements)

---

## Project Structure

```
bdd100k_object_detection/
├── README.md
├── docker-compose.yml
├── .gitignore
│
├── data_analysis/                # Task 1: Data Analysis
│   ├── Dockerfile               # Self-contained Docker image
│   ├── requirements.txt
│   ├── run_analysis.py          # Main analysis script
│   ├── dashboard.py             # Interactive Streamlit dashboard
│   └── src/
│       ├── __init__.py
│       ├── parser.py            # BDD100K JSON parser & data structures
│       ├── analyzer.py          # Statistical analysis functions
│       ├── visualizer.py        # Visualization utilities
│       └── utils.py             # Helper functions
│
├── model/                       # Task 2: Model Training
│   ├── requirements.txt
│   └── src/
│       ├── __init__.py
│       ├── dataset.py           # BDD100K → YOLO format converter
│       ├── model.py             # YOLO11 model wrapper & documentation
│       ├── train.py             # Training pipeline
│       └── inference.py         # Inference utilities
│
├── evaluation/                  # Task 3: Evaluation & Visualization
│   ├── requirements.txt
│   ├── run_evaluation.py        # Main evaluation script
│   └── src/
│       ├── __init__.py
│       ├── evaluate.py          # Metrics computation (mAP, AP, P/R)
│       ├── visualize.py         # Quantitative & qualitative viz
│       └── analysis.py          # Failure analysis & suggestions
│
├── notebooks/
│   └── BDD100K_Complete_Pipeline.ipynb  # Google Colab notebook
│
└── docs/
    ├── data_analysis_report.md
    ├── model_documentation.md
    └── evaluation_report.md
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Docker (for containerized data analysis)
- BDD100K dataset (labels + 100k images)

### Local Setup (without Docker)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/bdd100k_object_detection.git
cd bdd100k_object_detection

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install all dependencies
pip install -r data_analysis/requirements.txt
pip install -r model/requirements.txt
pip install -r evaluation/requirements.txt
```

### Dataset Preparation

Extract the BDD100K dataset so the structure looks like:
```
bdd100k/
├── images/
│   └── 100k/
│       ├── train/    (70,000 images)
│       └── val/      (10,000 images)
└── labels/
    ├── bdd100k_labels_images_train.json
    └── bdd100k_labels_images_val.json
```

---

## Task 1: Data Analysis

### Running with Docker

The data analysis is fully containerized and self-contained.

```bash
# Build the Docker image
cd data_analysis
docker build -t bdd100k-analysis .

# Run analysis (mount your data directory)
docker run -v /path/to/bdd100k:/data/bdd100k:ro \
           -v ./output:/output \
           bdd100k-analysis

# Run interactive dashboard
docker run -p 8501:8501 \
           -v /path/to/bdd100k:/data/bdd100k:ro \
           bdd100k-analysis \
           streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0 -- --data-root /data/bdd100k
```

Then open `http://localhost:8501` in your browser.

### Using Docker Compose

```bash
# Set the data path
export BDD_DATA_PATH=/path/to/bdd100k

# Run analysis
docker-compose up data-analysis

# Run dashboard
docker-compose up dashboard
```

### Running Locally

```bash
cd data_analysis

# Run full analysis
python run_analysis.py --data-root /path/to/bdd100k --output ./output

# Run interactive dashboard
streamlit run dashboard.py -- --data-root /path/to/bdd100k
```

### Analysis Outputs

The analysis produces:
- **Class distribution** charts (train & val)
- **Train vs Val comparison** showing distribution consistency
- **Bounding box statistics** (area, aspect ratio, dimensions per class)
- **Class co-occurrence** matrix
- **Spatial heatmaps** showing where objects appear in images
- **Attribute analysis** (weather, scene, time of day)
- **Anomaly detection** (tiny/huge boxes, extreme aspect ratios)
- **Interesting samples** identification
- **Occlusion/truncation** rates per class
- JSON report with all numerical results

### Key Findings (documented in `docs/data_analysis_report.md`)

---

## Task 2: Model

### Model Choice: YOLO11s (Ultralytics)

**Why YOLO11s:**

1. **State-of-the-art Performance** - Improved speed-accuracy trade-off over YOLOv8
2. **Anchor-free Design** - No anchor tuning needed, adapts to varied object sizes
3. **Multi-scale Detection** - P3/P4/P5 heads for small to large objects
4. **Strong Transfer Learning** - COCO pretraining covers most BDD100K classes
5. **Production Ready** - ONNX/TensorRT export, easy deployment
6. **Efficient Architecture** - 9.4M params (20% fewer than YOLOv8s) with better accuracy

**Architecture:**
- **Backbone**: CSPDarknet with C3k2 modules (efficient small-kernel CSP blocks)
- **Attention**: C2PSA (Cross Stage Partial with Spatial Attention) block
- **Neck**: PANet with C3k2 blocks for multi-scale feature fusion
- **Head**: Decoupled anchor-free detection head
- **Loss**: CIoU + BCE + DFL

See `model/src/model.py` for detailed architecture documentation.

### Training Pipeline

```bash
cd model

# Convert BDD100K to YOLO format and train (1 epoch demo)
python -m src.train \
    --bdd-root /path/to/bdd100k \
    --output runs \
    --model yolo11s \
    --epochs 1 \
    --batch-size 16 \
    --subset 1000

# Full training (recommended)
python -m src.train \
    --bdd-root /path/to/bdd100k \
    --output runs \
    --model yolo11s \
    --epochs 50 \
    --batch-size 16 \
    --device cuda
```

### Dataset Format Conversion

```bash
# Convert BDD100K labels to YOLO format
python -m src.dataset \
    --bdd-root /path/to/bdd100k \
    --output /path/to/yolo_dataset \
    --subset-size 1000  # Optional: use subset
```

### Inference

```bash
# Using trained weights
python -m src.inference \
    --model runs/train/bdd100k_yolo11s/weights/best.pt \
    --source /path/to/images \
    --output inference_results \
    --conf 0.25

# Or use a pretrained model name (auto-downloads)
python -m src.inference \
    --model yolo11m \
    --source /path/to/images \
    --output inference_results \
    --conf 0.25
```

---

## Task 3: Evaluation & Visualization

### Running Evaluation

Run from the **project root** (`bdd100k_object_detection/`):

```bash
python evaluation/run_evaluation.py \
    --model model/runs/train/bdd100k_yolo11s/weights/best.pt \
    --data-root /path/to/bdd100k \
    --output evaluation_results \
    --conf 0.25
```

Use the pretrained COCO model directly (auto-downloads if not present):

```bash
# Using model name (auto-downloads yolo11s.pt)
python evaluation/run_evaluation.py \
    --model yolo11s \
    --data-root /path/to/bdd100k \
    --output evaluation_results \
    --conf 0.25

# Or use a larger variant for better accuracy (auto-downloads yolo11m.pt)
python evaluation/run_evaluation.py \
    --model yolo11m \
    --data-root /path/to/bdd100k \
    --output evaluation_results \
    --conf 0.25
```

**Available model variants:** `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x` (larger = more accurate, slower)

### Metrics Used

| Metric | Rationale |
|--------|-----------|
| **mAP@0.5** | Standard benchmark metric, comparable to published results |
| **mAP@0.5:0.95** | COCO-style strict metric for localization quality |
| **Per-class AP** | Reveals class-specific weaknesses |
| **Precision/Recall** | Trade-off analysis at various confidence thresholds |
| **Confusion Matrix** | Shows systematic class confusions |
| **Size-based mAP** | Small/medium/large object performance breakdown |

### Qualitative Analysis

The evaluation produces:
- **Side-by-side GT vs Prediction** visualizations
- **Failure pattern clustering** (by size, position, attributes)
- **Error categorization** (classification, localization, false positives, missed)
- **Improvement suggestions** connecting data analysis findings

### Evaluation Outputs

```
evaluation_results/
├── plots/
│   ├── map_per_class.png
│   ├── performance_by_size.png
│   ├── error_analysis.png
│   └── precision_recall_*.png
├── qualitative/
│   ├── sample_0000.jpg  (GT vs Pred side-by-side)
│   └── ...
├── full_report.txt
└── results.json
```

---

## Docker Usage

### Data Analysis Container (Self-Contained)

```bash
# Build
docker build -t bdd100k-analysis ./data_analysis

# Run analysis (produces output in mounted volume)
docker run --rm \
    -v /path/to/bdd100k:/data/bdd100k:ro \
    -v $(pwd)/output:/output \
    bdd100k-analysis

# Run dashboard (interactive)
docker run --rm -p 8501:8501 \
    -v /path/to/bdd100k:/data/bdd100k:ro \
    bdd100k-analysis \
    streamlit run dashboard.py \
        --server.port 8501 \
        --server.address 0.0.0.0 \
        -- --data-root /data/bdd100k
```

**Note:** The Docker image is completely self-contained. No additional installations are needed on the host system beyond Docker itself.

### Windows Docker Commands

```powershell
# Build
docker build -t bdd100k-analysis .\data_analysis

# Run analysis
docker run --rm `
    -v C:\path\to\bdd100k:/data/bdd100k:ro `
    -v ${PWD}\output:/output `
    bdd100k-analysis

# Run dashboard
docker run --rm -p 8501:8501 `
    -v C:\path\to\bdd100k:/data/bdd100k:ro `
    bdd100k-analysis `
    streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0 -- --data-root /data/bdd100k
```


## Requirements

### System Requirements
- Python 3.10+
- Docker (for containerized analysis)
- 8GB+ RAM
- GPU recommended for training

### Python Dependencies

Core libraries used:
- `ultralytics` - YOLO11 framework
- `torch` / `torchvision` - Deep learning backend
- `numpy` / `pandas` - Numerical computing
- `matplotlib` / `seaborn` - Static visualizations
- `opencv-python` - Image processing
- `streamlit` - Interactive dashboard


## License

This project is for educational/assessment purposes
