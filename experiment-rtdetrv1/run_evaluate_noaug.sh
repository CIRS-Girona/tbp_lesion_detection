#!/bin/bash
# ============================================================
# run_evaluate_noaug.sh
# Threshold sweep evaluation for RT-DETR-L -- NO AUG model
# Run from: ~/code/iToBoS/
# ============================================================

set -e

DEVICE="${1:-0}"
WANDB_API_KEY="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"

cd ~/code/iToBoS
source ~/code/iToBoS/.venv/bin/activate
export WANDB_API_KEY="${WANDB_API_KEY}"

echo "=============================================="
echo "  RT-DETR-L Evaluation -- NO AUG (GPU ${DEVICE})"
echo "=============================================="

python praveen/experiment-rtdetr/evaluate.py \
    --weights praveen/experiment-rtdetr/runs/best_model/rtdetr-l_best_noaug_100ep/weights/best.pt \
    --run_name rtdetr-l_noaug_evaluation \
    --name noaug \
    --device "${DEVICE}"

echo ""
echo "  Done! Results in:"
echo "    praveen/experiment-rtdetr/runs/evaluation/noaug/"
