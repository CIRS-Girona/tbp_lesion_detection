#!/bin/bash
# ============================================================
# run_train_noaug.sh
# Full 100-epoch training for RT-DETR-L -- NO AUGMENTATION
# Loads best HP config from: best_sweep_config_rtdetr-l_noaug.json
# Run from: ~/code/iToBoS/
# ============================================================

set -e

DEVICE="${1:-0}"
WANDB_API_KEY="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"

cd ~/code/iToBoS
source ~/code/iToBoS/.venv/bin/activate
export WANDB_API_KEY="${WANDB_API_KEY}"

echo "=============================================="
echo "  RT-DETR-L Full Training -- NO AUG (GPU ${DEVICE})"
echo "  Output: praveen/experiment-rtdetr/runs/best_model/"
echo "=============================================="

python praveen/experiment-rtdetr/train_best.py --aug_mode noaug --device "${DEVICE}"

echo ""
echo "  Done! Weights at:"
echo "    praveen/experiment-rtdetr/runs/best_model/rtdetr-l_best_noaug_100ep/weights/best.pt"
echo "  Next: bash praveen/experiment-rtdetr/run_evaluate_noaug.sh"
