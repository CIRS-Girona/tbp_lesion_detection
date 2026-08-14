# MSDETR Setup and Usage Guide

## Environment Setup

### 1. Create Environment
```bash
conda create -n MSDETR python=3.12
```

### 2. Activate Environment
```bash
conda activate MSDETR
```

### 3. Install Packages and Dependencies
```bash
pip install -e .
pip install -r requirements.txt
```

## Dataset Configuration

### 4. Prepare Dataset

Before training, you need to prepare your dataset first. Please refer to `datasetConfigTemplate.yaml` for dataset configuration and path settings.

**Steps:**
- Download or prepare your dataset
- Organize the dataset according to the required structure
- Configure dataset paths in `datasetConfigTemplate.yaml`

## Model Training and Validation

### 5. Training
```bash
python tools/train.py
```

### 6. Validation
```bash
python tools/val.py
```

---

## Quick Start

Complete installation process:
```bash
# Create and activate environment
conda create -n MSDETR python=3.12
conda activate MSDETR

# Install dependencies
pip install -e.
pip install -r requirements.txt

# Prepare and configure dataset (refer to datasetConfigTemplate.yaml)

# Start training
python tools/train.py

# Model validation
python tools/val.py
```

## Notes

- **Dataset must be prepared and configured before training**
- Dataset paths must be correctly configured in `datasetConfigTemplate.yaml`
- Please confirm GPU environment is properly configured before training