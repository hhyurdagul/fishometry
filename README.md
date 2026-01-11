# Fishometry

Fish morphometry analysis using computer vision and machine learning.

## Project Structure

```
src/
├── preprocessing/          # Image preprocessing pipeline
│   ├── run.py              # Main preprocessing entry point
│   └── steps/              # Processing steps (YOLO, rotate, depth, etc.)
├── data/                   # Dataset preparation
│   ├── split.py            # Split raw.csv into train/val/test
│   └── augment.py          # Zoom augmentation
├── training/               # Model training
│   ├── run.py              # Training orchestrator
│   ├── data_loader.py      # Shared data loading utilities
│   ├── baseline.py         # Baseline model
│   ├── regression.py       # Linear regression
│   ├── xgboost.py          # XGBoost model
│   ├── mlp.py              # MLP neural network
│   ├── cnn.py              # CNN model
│   └── per_fishtype.py     # Per-fish-type models
├── visualization/          # Streamlit dashboard
│   └── app.py
└── utils/
    └── io.py               # Config/image/CSV I/O utilities
```

## Setup

```bash
# Install dependencies
uv sync
```

## Usage

### 1. Data Preparation

```bash
# Split dataset into train/val/test
python -m src.data.split --input data/data-inside/raw.csv --output data/data-inside/splits

# Apply zoom augmentation (optional)
python -m src.data.augment --source data-inside --dest data-inside-zoom
```

### 2. Preprocessing

```bash
# Run preprocessing pipeline
python -m src.preprocessing.run --config configs/config_inside.yaml --all-splits
```

### 3. Training

```bash
# Baseline model
python -m src.training.baseline --dataset data-inside

# Linear regression
python -m src.training.regression --dataset data-inside --feature-set coords

# XGBoost
python -m src.training.xgboost --dataset data-inside --feature-set coords

# MLP
python -m src.training.mlp --dataset data-inside --feature-set coords --epochs 200

# CNN
python -m src.training.cnn --dataset data-inside --feature-set coords --epochs 100

# Per-fish-type models
python -m src.training.per_fishtype --dataset data-outside --feature-set scaled --depth

# Run full pipeline (preprocessing + training)
python -m src.training.run --pipeline 1 --dataset data-inside
```

### 4. Visualization

```bash
streamlit run src/visualization/app.py
```

## Configurations

- `configs/config_inside.yaml` - Inside dataset configuration
- `configs/config_inside_zoom.yaml` - Zoomed inside dataset
- `configs/config_outside.yaml` - Outside dataset configuration
