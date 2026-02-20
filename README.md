# LEXAI: Explainable AI for Leukemia Detection

A dual-pathway deep learning system combining **CNN Global Analysis** and **GNN Spatial Analysis** for transparent, explainable leukemia subtype classification from blood smear images.

## Architecture

```
Input Image
  ├→ CNN Ensemble (EfficientNet + ResNet50 + DenseNet121) → 512-dim global features
  └→ Cell Segmentation → Graph Construction → GNN (GCN→GAT→GraphSAGE) → 256-dim spatial features
    ↓
  Multi-Modal Fusion (Cross-modal Attention) → 512-dim fused features
    ↓
  Multi-Task Outputs:
    1. Classification: ALL Blast / ALL Early Pre-B / ALL Pre-B / ALL Pro-B / Benign
    2. Spatial Pattern Score
    3. Blast Percentage
    4. Explainability: Grad-CAM heatmaps + GNN Attention visualization
    5. Uncertainty: MC Dropout confidence intervals
```

## Key Features

- **5-Class Leukemia Subtype Classification** — not just binary, but specific ALL subtypes
- **Explainable AI** — Grad-CAM heatmaps show *where* the model looks, GNN attention shows *which cells matter*
- **Uncertainty Estimation** — MC Dropout provides confidence intervals for clinical safety
- **Anti-Overfitting Pipeline** — RandomResizedCrop, GaussianBlur, RandomErasing, WeightedRandomSampler, backbone freezing, differential LR, class-weighted loss
- **Stratified K-Fold CV** — robust evaluation across the full dataset
- **Interactive Web Dashboard** — upload images and get instant results with visualizations

## Datasets

The training data is **not included** in this repository. Download the following datasets from Kaggle and run the organizer script.

### Required Datasets

| Dataset | Images | Kaggle Link |
|---|---|---|
| **C-NMC 2019** (ALL blast vs Normal) | ~15,000 | [andrewmvd/leukemia-classification](https://www.kaggle.com/datasets/andrewmvd/leukemia-classification) |
| **ALL Image Dataset** (subtypes) | ~3,200 | [mehradaria/leukemia](https://www.kaggle.com/datasets/mehradaria/leukemia) |
| **Blood Cell Cancer [ALL]** (subtypes) | ~3,200 | [mohammadamireshraghi/blood-cell-cancer-all-4class](https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class) |

### Optional AML Datasets (require Kaggle access approval)

| Dataset | Images | Kaggle Link |
|---|---|---|
| AML Cytomorphology (LMU) | ~18,000 | [gustaveroussy/aml-cytomorphology](https://www.kaggle.com/datasets/gustaveroussy/aml-cytomorphology) |
| Blood Cancer Image Dataset | ~10,000 | [samuelcortinhas/blood-cancer-image-dataset](https://www.kaggle.com/datasets/samuelcortinhas/blood-cancer-image-dataset) |

### Setup Data

```bash
# 1. Install Kaggle CLI and set up credentials
pip install kaggle
# Place your kaggle.json in ~/.kaggle/

# 2. Download datasets
python scripts/download_dataset.py

# 3. Organize into class folders
python scripts/reorganize_dataset.py
```

After running, your `data/` directory should look like:
```
data/
├── ALL_Blast/          (11,725 images)
├── ALL_Early_Pre_B/    ( 1,964 images)
├── ALL_Pre_B/          ( 1,918 images)
├── ALL_Pro_B/          ( 1,600 images)
└── Benign/             ( 4,405 images)
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

## Training

```bash
# Standard training (with all anti-overfitting measures)
python scripts/train.py --data_dir data --epochs 50 --batch_size 8 --device cuda

# Stratified 5-Fold Cross-Validation (recommended)
python scripts/train_kfold.py --data_dir data --k 5 --epochs 50 --batch_size 8 --device cuda
```

## Web Dashboard

```bash
# Start the server (loads model from checkpoints/best_model.pth)
python -m api.server

# Open http://localhost:8000 in your browser
```

## Project Structure

```
LEXAI/
├── lexai/                     # ML pipeline
│   ├── config.py              # Central configuration
│   ├── data/                  # Dataset, preprocessing, segmentation
│   ├── models/                # CNN ensemble, GNN pathway, fusion, LEXAI model
│   ├── explainability/        # Grad-CAM, GNN attention visualization
│   ├── uncertainty/           # MC Dropout uncertainty estimation
│   └── training/              # Trainer, losses, metrics
├── api/                       # FastAPI backend
│   ├── server.py              # API server
│   ├── inference.py           # Inference pipeline
│   └── schemas.py             # Response models
├── web/                       # Frontend dashboard
│   ├── index.html
│   ├── index.css
│   └── index.js
├── scripts/                   # Training & utility scripts
│   ├── train.py               # Standard training
│   ├── train_kfold.py         # K-Fold cross-validation
│   ├── download_dataset.py    # Dataset downloader
│   ├── reorganize_dataset.py  # Dataset organizer
│   └── generate_demo.py       # Synthetic data generator
└── tests/                     # Unit & integration tests
```

## Requirements

- Python 3.10+
- PyTorch 2.x (CUDA recommended)
- NVIDIA GPU with 8GB+ VRAM (for training)

## Team

**University of Petroleum and Energy Studies**
- Garima Aishwarya, Brajraj Singh Pathania, Ayushmaan Singh, Piyush Bharadwaj
- Guide: Prof. Gouranga Duari

## License

This project is for academic research purposes.
