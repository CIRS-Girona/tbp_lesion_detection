#!/bin/bash
# ============================================================
# run_evaluate_full.sh
# Threshold sweep evaluation for RT-DETR-L -- FULL AUG model
# Run from: ~/code/iToBoS/
# ============================================================

set -e

DEVICE="${1:-0}"
WANDB_API_KEY="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"

cd ~/code/iToBoS
source ~/code/iToBoS/.venv/bin/activate
export WANDB_API_KEY="${WANDB_API_KEY}"

echo "=============================================="
echo "  RT-DETR-L Evaluation -- FULL AUG (GPU ${DEVICE})"
echo "=============================================="

python praveen/experiment-rtdetr/evaluate.py \
    --weights praveen/experiment-rtdetr/runs/best_model/rtdetr-l_best_full_100ep/weights/best.pt \
    --run_name rtdetr-l_full_evaluation \
    --name full \
    --device "${DEVICE}"

echo ""
echo "  Done! Results in:"
echo "    praveen/experiment-rtdetr/runs/evaluation/full/"
