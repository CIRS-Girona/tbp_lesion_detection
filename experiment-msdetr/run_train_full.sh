#!/usr/bin/env bash
# run_train_full.sh — MS-DETR 100-epoch training, full augmentation
# Run from ~/code/iToBoS/ AFTER run_sweep_full.sh completes:
#   bash yamin/experiment-msdetr/run_train_full.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$REPO_ROOT/yamin/msdetr-venv/bin/activate"
[[ -f "$VENV" ]] || { echo "[ERROR] msdetr-venv not found."; exit 1; }
source "$VENV"
cd "$REPO_ROOT"

CONFIG="yamin/experiment-msdetr/best_sweep_config_msdetr_full.json"
[[ -f "$CONFIG" ]] || { echo "[ERROR] Sweep config not found: $CONFIG — run run_sweep_full.sh first."; exit 1; }

echo "[$(date '+%H:%M:%S')] Training MS-DETR — full augmentation (100 epochs)"
CUDA_VISIBLE_DEVICES=1 python yamin/experiment-msdetr/train_best.py --aug_mode full

WEIGHTS="yamin/experiment-msdetr/runs/best_model/msdetr_best_full_100ep/weights/best.pt"
[[ -f "$WEIGHTS" ]] && echo "[$(date '+%H:%M:%S')] Done. Weights: $WEIGHTS" \
    || echo "[WARN] Weights not found at $WEIGHTS — check training output."
