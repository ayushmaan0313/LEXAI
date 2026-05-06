<div align="center">

# LEXAI

### Explainable AI for Leukemia Detection

A dual-pathway deep learning system combining **CNN ensemble analysis** and **GNN spatial reasoning** for transparent, explainable leukemia subtype classification from peripheral blood smear microscopy.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![License: Academic](https://img.shields.io/badge/License-Academic-lightgrey)](#license)

```
Normal  ·  ALL (Acute Lymphoblastic)  ·  AML (Acute Myeloid)  ·  CML (Chronic Myeloid)
```

---

</div>

## Architecture

```
                              ┌─────────────────────────────┐
                              │      Input Blood Smear      │
                              └──────────┬──────────────────┘
                                         │
                    ┌────────────────────┤├────────────────────┐
                    ▼                                          ▼
        ┌───────────────────────┐                 ┌──────────────────────┐
        │     CNN Ensemble      │                 │   Cell Segmentation  │
        │  ┌─────────────────┐  │                 │  (Watershed + Otsu)  │
        │  │ EfficientNet-B0 │  │                 └──────────┬───────────┘
        │  │ ResNet-50       │  │                            ▼
        │  │ DenseNet-121    │  │                 ┌──────────────────────┐
        │  │ ViT-B/16        │  │                 │   Graph Construction │
        │  └─────┬───────────┘  │                 │   (k-NN, k=6)       │
        │        ▼              │                 └──────────┬───────────┘
        │  Learnable Fusion     │                            ▼
        │  (Softmax Weights)    │                 ┌──────────────────────┐
        └───────────┬───────────┘                 │    GNN Pathway       │
                    │                             │  GCN → GAT → SAGE   │
                    │ 512-dim                     └──────────┬───────────┘
                    │                                        │ 256-dim
                    └────────────────┬┬──────────────────────┘
                                     ▼
                        ┌────────────────────────┐
                        │  Cross-Modal Attention  │
                        │  (8-head, 512-dim out)  │
                        └────────────┬───────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │Classification│ │Spatial Score │ │ Blast Density│
            │ 4 classes    │ │ Pattern map  │ │  Percentage  │
            └──────────────┘ └──────────────┘ └──────────────┘
```

## Key Features

| Feature | Description |
|---|---|
| **4-Class Classification** | Normal / ALL / AML / CML subtype detection from single cell crops or full-field smears |
| **Explainable AI** | Grad-CAM heatmaps show *where* the model focuses; GNN attention reveals *which cells matter* |
| **Uncertainty Estimation** | Monte Carlo Dropout provides confidence intervals for clinical safety |
| **Temperature Calibration** | Post-hoc LBFGS calibration targets ECE < 0.05 for reliable probability outputs |
| **Per-Cell Analysis** | Full blood smear images are segmented, each cell classified individually, votes aggregated |
| **Mixed Precision Training** | AMP (fp16) for ~2x speedup and ~50% VRAM reduction on consumer GPUs |
| **3-Stage Training** | Freeze → Fine-tune → Calibrate pipeline with early stopping and cosine annealing |
| **React Dashboard** | Upload images, view Grad-CAM overlays, batch analysis with CSV/JSON export |

## Tech Stack

<table>
<tr>
<td width="50%">

### ML & Training
- **PyTorch 2.x** — model backbone + AMP training
- **torchvision** — pretrained CNN weights
- **torch-geometric** — GNN layers (GCN, GAT, GraphSAGE)
- **efficientnet-pytorch** — EfficientNet-B0 backbone
- **scikit-learn** — metrics, stratified splits
- **OpenCV** — cell segmentation, stain normalization

</td>
<td width="50%">

### Web & API
- **FastAPI** + Uvicorn — REST API backend
- **React 19** + Vite — SPA frontend
- **Tailwind CSS v4** — utility-first styling
- **Recharts** — training curve visualization
- **Lucide React** — iconography

</td>
</tr>
</table>

## Datasets

Training data is **not included** in this repository. The download scripts handle everything automatically.

| Dataset | Source | Images | Classes |
|---|---|---|---|
| **C-NMC 2019** | [Kaggle](https://www.kaggle.com/datasets/andrewmvd/leukemia-classification) | ~15,000 | ALL blast vs Normal |
| **Blood Cell Cancer** | [Kaggle](https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class) | ~3,200 | ALL subtypes |
| **ALL-IDB** | [Kaggle](https://www.kaggle.com/datasets/mehradaria/leukemia) | ~3,200 | ALL subtypes |
| **AML Cytomorphology (LMU)** | [TCIA](https://www.cancerimagingarchive.net/) | ~18,000 | AML cell types |

## Getting Started

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 8 GB+ VRAM (training) or CPU-only (inference)
- [Kaggle API credentials](https://www.kaggle.com/settings) (`~/.kaggle/kaggle.json`)

### 1. Clone & Install

```bash
git clone https://github.com/ayushmaan0313/LEXAI.git
cd LEXAI
pip install -r requirements.txt
```

### 2. Download Datasets

```bash
# Download all datasets (~5-10 GB)
python scripts/download_dataset.py --all --output_dir data

# Balance classes and generate train/val/test manifests
python scripts/prepare_data.py --data_dir data --balance --strategy hybrid
```

After preparation, `data/manifests/` will contain `train.csv`, `val.csv`, and `test.csv`.

### 3. Train

**Option A — Local GPU (RTX 3060+ / 8 GB VRAM)**

```bash
python scripts/train.py \
  --data_dir data \
  --manifest data/manifests/train.csv \
  --epochs 40 \
  --batch_size 16 \
  --lr 3e-4 \
  --finetune_lr 5e-5 \
  --device cuda
```

**Option B — Google Colab (recommended for free T4 GPU)**

Open `notebooks/train_colab.ipynb` in Colab, upload your project as a zip, and follow the cells. Checkpoints save to Google Drive automatically.

### 4. Run the Dashboard

```bash
# Build frontend
cd web && npm install && npm run build && cd ..

# Start server (loads best checkpoint)
python -m api.server
```

Open **http://localhost:8000** — upload a blood smear image and get instant classification with Grad-CAM overlays.

## Training Pipeline

The trainer uses a **3-stage pipeline** with automatic transitions:

```
Stage 1 — Frozen Backbones (10 epochs)
│  Only classification head, fusion weights, and GNN train
│  LR: 3e-4 · Cosine annealing · Early stopping (patience=15)
│
Stage 2 — Full Fine-Tuning (20 epochs)
│  All parameters unfrozen · Low LR: 5e-5
│  Fresh GradScaler (init_scale=1024) to prevent fp16 overflow
│  Cosine annealing · Early stopping (patience=15)
│
Stage 3 — Temperature Calibration
   LBFGS optimizer fits a single temperature scalar on validation set
   Target: ECE < 0.05
```

**Anti-overfitting measures:** label smoothing (0.1), class-weighted cross-entropy, gradient clipping (max_norm=1.0), dropout (0.3–0.5), data augmentation (random crop, flip, color jitter, Gaussian blur, random erasing).

## Project Structure

```
LEXAI/
├── lexai/                          # Core ML package
│   ├── config.py                   # All hyperparameters and defaults
│   ├── data/
│   │   ├── dataset.py              # DataLoader creation, manifest loading
│   │   ├── preprocessing.py        # Transforms, stain normalization
│   │   └── segmentation.py         # Watershed cell segmentation
│   ├── models/
│   │   ├── lexai_model.py          # Full dual-pathway model
│   │   ├── cnn_ensemble.py         # Weighted CNN fusion + calibration
│   │   ├── cnn_backbone.py         # Individual backbone wrappers
│   │   ├── gnn_pathway.py          # GCN → GAT → GraphSAGE pipeline
│   │   └── fusion.py               # Cross-modal attention fusion
│   ├── training/
│   │   ├── trainer.py              # 3-stage training loop with AMP
│   │   ├── losses.py               # Multi-task loss (CE + MSE)
│   │   └── metrics.py              # Accuracy, ECE, FAR, EER
│   ├── explainability/
│   │   └── gradcam.py              # Grad-CAM heatmap generation
│   └── uncertainty/
│       └── mc_dropout.py           # Monte Carlo Dropout inference
│
├── api/                            # FastAPI backend
│   ├── server.py                   # REST endpoints + static file serving
│   ├── inference.py                # Image → prediction pipeline
│   └── schemas.py                  # Pydantic response models
│
├── web/                            # React frontend (Vite + Tailwind v4)
│   └── src/
│       ├── App.jsx                 # Main app shell
│       ├── api.js                  # API client
│       └── components/
│           ├── AnalyzePage.jsx     # Single image analysis + Grad-CAM
│           ├── BatchAnalysis.jsx   # Multi-image batch processing
│           ├── BackboneComparison.jsx  # Per-backbone feature comparison
│           └── TrainingDashboard.jsx   # Live training curves
│
├── scripts/
│   ├── train.py                    # Training entrypoint
│   ├── train_kfold.py             # Stratified K-Fold cross-validation
│   ├── download_dataset.py         # Kaggle dataset downloader
│   ├── download_tcia.py            # TCIA AML dataset downloader
│   ├── prepare_data.py             # Class balancing + manifest generation
│   └── generate_demo.py            # Synthetic demo data generator
│
├── notebooks/
│   └── train_colab.ipynb           # Google Colab training notebook
│
└── tests/
    ├── test_models.py              # Model architecture tests
    └── test_pipeline.py            # End-to-end pipeline tests
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/analyze` | Classify a single blood smear image |
| `GET` | `/api/health` | Server health check |
| `GET` | `/api/config` | Current model configuration |
| `GET` | `/*` | React SPA (served from `web/dist/`) |

**`POST /api/analyze`** accepts `multipart/form-data` with an `image` field and returns:

```json
{
  "predicted_class": "ALL",
  "confidence": 0.9412,
  "probabilities": { "Normal": 0.02, "ALL": 0.94, "AML": 0.03, "CML": 0.01 },
  "is_uncertain": false,
  "cell_count": 47,
  "blast_percentage": 68.2,
  "gradcam_overlay": "<base64 PNG>",
  "cell_analysis": { "per_cell_results": [...], "vote_distribution": {...} }
}
```

## Team

**University of Petroleum and Energy Studies**

- Garima Aishwarya, Brajraj Singh Pathania, Ayushmaan Singh, Piyush Bharadwaj
- Guide: Prof. Gouranga Duari

## License

This project is for academic and research purposes.
