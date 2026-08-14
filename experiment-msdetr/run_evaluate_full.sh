#!/usr/bin/env bash
# run_evaluate_full.sh — MS-DETR threshold sweep evaluation, full augmentation
# Run from ~/code/iToBoS/ AFTER run_train_full.sh completes:
#   bash yamin/experiment-msdetr/run_evaluate_full.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$REPO_ROOT/yamin/msdetr-venv/bin/activate"
[[ -f "$VENV" ]] || { echo "[ERROR] msdetr-venv not found."; exit 1; }
source "$VENV"
cd "$REPO_ROOT"

WEIGHTS="yamin/experiment-msdetr/runs/best_model/msdetr_best_full_100ep/weights/best.pt"
[[ -f "$WEIGHTS" ]] || { echo "[ERROR] Weights not found: $WEIGHTS — run run_train_full.sh first."; exit 1; }

echo "[$(date '+%H:%M:%S')] Evaluating MS-DETR — full augmentation (11x9 threshold sweep)"
CUDA_VISIBLE_DEVICES=1 python yamin/experiment-msdetr/evaluate.py \
    --weights "$WEIGHTS" \
    --name    full
echo "[$(date '+%H:%M:%S')] Evaluation (full) complete."
echo "  CSV results: yamin/experiment-msdetr/runs/evaluation/full/threshold_sweep_test.csv"
