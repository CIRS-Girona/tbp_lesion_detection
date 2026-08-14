# TBP Lesion Detection: Multi-Architecture Benchmark

Code for the paper **"Automated Skin Lesion Detection in Total Body Photography: A Multi-Architecture Benchmark"**, accepted as a spotlight paper at ISIC Workshop, MICCAI 2026.

We benchmark seven real-time object detectors on the [iToBoS dataset](https://itobos.eu/) for skin lesion detection in total body photography (TBP). Rather than a simple mAP leaderboard, we provide a mechanistic analysis covering error decomposition (TIDE), confidence regime characterisation, EigenCAM saliency, and augmentation ablations.

---

## Results Summary

| Model | Aug | F1 | mAP50 | P | R | GPU FPS |
|---|---|---|---|---|---|---|
| **YOLOv12s** | Full | **0.640** | 0.652 | 0.680 | 0.604 | 35.2 |
| D-FINE-S | Full | 0.639 | **0.674** | **0.707** | 0.583 | 67.8 |
| YOLOv26s | Full | 0.634 | 0.644 | 0.680 | 0.593 | **46.5** |
| RT-DETRv2-S | Full | 0.633 | 0.668 | 0.703 | 0.576 | 42.9 |
| YOLOv11s | Full | 0.631 | 0.643 | 0.657 | 0.606 | 61.3 |
| YOLOv8s | Full | 0.627 | 0.622 | 0.629 | 0.625 | 84.2 |
| MS-DETR | Full | 0.614 | 0.642 | 0.643 | 0.588 | 33.8 |

> **D-FINE-S** is our primary recommendation: highest mAP50, highest precision, smallest augmentation dependence, and lesion-focused EigenCAM activations. All models are evaluated at 1024×1024px on an NVIDIA A100 80GB.

---

## Repository Structure

```
tbp_lesion_detection/
├── experiment-v8/          # YOLOv8s experiments (Kaggle challenge baseline)
├── experiment-v11/         # YOLOv11s experiments
├── experiment-v12/         # YOLOv12s experiments (best F1)
├── experiment-v26/         # YOLOv26s experiments (NMS-free)
├── experiment-solomon/     # Solomon et al. (2024) Kaggle-winning baseline reproduction
├── experiment-msdetr/      # MS-DETR experiments
├── experiment-rtdetrv1/    # RT-DETR-L via Ultralytics + LesionSCALE modules
├── experiment-rtdetrv2/    # RT-DETRv2-S (official Baidu repo, Optuna sweep)
├── experiment-dfine/       # D-FINE-S experiments + ablations, TIDE, saliency, uncertainty
└── benchmark-gflops/       # GPU/CPU speed and GFLOPs benchmarks
```

Each experiment folder follows the same three-step pipeline:
1. `hparam_sweep.py` — Bayesian HP sweep (W&B) to find optimal training config
2. `train_best.py` — Full 100-epoch training with the best sweep config
3. `evaluate.py` — Threshold sweep evaluation (11 conf × 9 NMS-IoU grid)

---

## Setup

### Requirements

- Python 3.10+
- PyTorch 2.x with CUDA
- NVIDIA GPU (experiments run on A100 80GB)

### Install

```bash
# Clone the repo
git clone https://github.com/your-org/tbp_lesion_detection.git
cd tbp_lesion_detection

# Install YOLO experiments (v8, v11, v12, v26, solomon, msdetr)
pip install ultralytics wandb

# For D-FINE experiments, clone D-FINE separately:
git clone https://github.com/Peterande/D-FINE.git
cd D-FINE && pip install -r requirements.txt

# For RT-DETRv2 experiments:
cd experiment-rtdetrv2
pip install -r requirements.txt
```

### Environment Variables

Set these before running any script:

```bash
# Your W&B credentials (https://wandb.ai/settings)
export WANDB_ENTITY="your-wandb-entity"
export WANDB_API_KEY="your-api-key"

# Path to the iToBoS dataset (split_dataset_blurred layout)
export DATA_ROOT="/path/to/itobos/split_dataset_blurred"

# For D-FINE scripts: path to the COCO-format dataset
export DFINE_DATASET_DIR="/path/to/dfine_results/dataset_coco"
```

### Dataset

Download the [iToBoS dataset](https://itobos.eu/) and place it so that:
```
$DATA_ROOT/
├── train/images/
├── val/images/
├── test/images/
└── annotations/
    ├── train.json
    ├── val.json
    └── test.json
```

A `params.yaml` file pointing to the dataset is required for YOLO scripts — place it in your project root:
```yaml
path: /path/to/itobos/split_dataset_blurred
train: train/images
val: val/images
test: test/images
nc: 1
names: ['lesion']
```

---

## Running Experiments

### YOLO Variants (v8, v11, v12, v26)

All four follow the same workflow. Example for YOLOv12s:

```bash
cd experiment-v12

# Step 1: Bayesian HP sweep (25 trials, 40 epochs each)
wandb sweep sweep_config_v12_full.yaml        # prints sweep ID
wandb agent <entity>/<project>/<sweep-id>     # run the sweep

# Step 2: Full 100-epoch training with best config
python train_best.py --aug_mode full          # full augmentation
python train_best.py --aug_mode noaug         # no-augmentation ablation

# Step 3: Evaluate on test set (threshold sweep)
python evaluate.py --aug_mode full
```

Shell scripts are provided for running on a SLURM/SSH cluster:
```bash
bash run_sweep_full.sh
bash run_train_full.sh
bash run_evaluate_full.sh
```

### Solomon Baseline

Reproduces the iToBoS Kaggle challenge-winning YOLOv8s:

```bash
cd experiment-solomon
python train.py --aug_mode full
python evaluate.py --aug_mode full
```

### MS-DETR

Uses a patched Ultralytics fork (included in `source-code-ms-detr/`):

```bash
cd experiment-msdetr
pip install -e source-code-ms-detr/   # install the MS-DETR fork

# Then run sweep/train/evaluate as usual
python hparam_sweep.py --aug_mode full
python train_best.py --aug_mode full
python evaluate.py --aug_mode full
```

### RT-DETRv2-S (Official Baidu Repo)

```bash
cd experiment-rtdetrv2
pip install -r requirements.txt

# HP sweep (Optuna, SQLite backend for multi-GPU parallelism)
python hp_sweep_rtdetrv2.py

# Training (uses torchrun for distributed)
torchrun --nproc_per_node=1 tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_itobos_1024.yml \
  --use-amp --seed 0

# Evaluation
python eval_f1_pr.py --checkpoint /path/to/checkpoint.pth
```

### D-FINE-S

```bash
cd experiment-dfine

# HP sweep
python sweep_agent.py

# Full training (run from the D-FINE repo root)
bash run_02_train_fullaug.sh
bash run_03_train_noaug.sh

# Loss ablations
bash run_04_ablation_losses.sh

# Full evaluation pipeline
bash run_05_evaluate_all.sh

# TIDE error decomposition
python tide_analysis.py --pred /path/to/predictions.json

# EigenCAM saliency maps
python saliency_analysis.py --mode eigencam --model dfine

# Matched saliency (same images for D-FINE and YOLO comparison)
python saliency_matched.py

# Uncertainty distribution plots
python uncertainty_distribution.py

# Multi-threshold mAP (0.50 to 0.95)
python eval_map_multi_threshold.py
```

### LesionSCALE Module (RT-DETR ablation)

`experiment-rtdetrv1/` contains the LesionSCALE architecture modules — three enhancement blocks for the RT-DETR decoder:
- **DSA**: Dynamic scale aggregation cross-attention
- **FAR**: Frequency-domain FFT filtering for low-contrast lesions  
- **SARE**: Strip-pooling for irregular border enhancement

```bash
cd experiment-rtdetrv1
python test_modules.py      # unit tests (shape, gradients, identity-at-init)
python verify_scale.py      # confirm modules are active in decoder
python train_scale.py --aug_mode full
```

---

## Benchmarks

GPU/CPU latency and GFLOPs for all models:

```bash
cd benchmark-gflops

# GPU latency for YOLO models (20 warmup + 100 timed runs)
python benchmark_fps.py

# CPU latency (10 warmup + 50 timed runs)
python benchmark_yolo_msdetr_cpu.py
python benchmark_dfine_cpu.py
python benchmark_rtdetrv2_cpu.py

# GFLOPs (MACs)
python benchmark_yolo_msdetr_gflops.py   # at 640px
python benchmark_dfine_gflops.py          # at 1024px
python benchmark_rtdetrv2_gflops.py       # at 1024px
```

---

## Key Findings

1. **Confidence operating regimes are architectural, not tunable**: YOLO models plateau at conf=0.20; RT-DETRv2-S and D-FINE-S work at conf=0.50; MS-DETR is completely threshold-invariant.
2. **NMS-free doesn't help accuracy in TBP**: Lesions rarely overlap, so NMS is never triggered. The NMS-free advantage of YOLOv26s surfaces only as a speed gain.
3. **D-FINE-S's distribution loss acts as implicit regularisation**: Its augmentation delta is only +0.17pp F1 vs +0.8–1.9pp for all others.
4. **EigenCAM reveals a Clever Hans effect in YOLOv12s**: Activations focus on hair texture rather than lesion morphology — a deployment risk on hairless body sites.
5. **Recall is the bottleneck**: At any deployment threshold, 37–43% of annotated lesions are missed. This is a data-scale limitation, not an architecture one.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{yamin2026tbplesion,
  title     = {Automated Skin Lesion Detection in Total Body Photography: A Multi-Architecture Benchmark},
  author    = {Yamin, Muhammad and Murali, Praveen Kumar and Das, Aritra and Wahid, Sheikh Abdul},
  booktitle = {ISIC Workshop, MICCAI},
  year      = {2026}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
