#!/usr/bin/env bash
# run_evaluate_noaug.sh — MS-DETR threshold sweep evaluation, no augmentation
# Run from ~/code/iToBoS/ AFTER run_train_noaug.sh completes:
#   bash yamin/experiment-msdetr/run_evaluate_noaug.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$REPO_ROOT/yamin/msdetr-venv/bin/activate"
[[ -f "$VENV" ]] || { echo "[ERROR] msdetr-venv not found."; exit 1; }
source "$VENV"
cd "$REPO_ROOT"

WEIGHTS="yamin/experiment-msdetr/runs/best_model/msdetr_best_noaug_100ep/weights/best.pt"
[[ -f "$WEIGHTS" ]] || { echo "[ERROR] Weights not found: $WEIGHTS — run run_train_noaug.sh first."; exit 1; }

echo "[$(date '+%H:%M:%S')] Evaluating MS-DETR — no augmentation (11x9 threshold sweep)"
CUDA_VISIBLE_DEVICES=1 python yamin/experiment-msdetr/evaluate.py \
    --weights "$WEIGHTS" \
    --name    noaug
echo "[$(date '+%H:%M:%S')] Evaluation (noaug) complete."
echo "  CSV results: yamin/experiment-msdetr/runs/evaluation/noaug/threshold_sweep_test.csv"
