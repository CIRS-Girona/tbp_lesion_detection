#!/bin/bash
# ============================================================
# run_train_full.sh
# Full 100-epoch training for YOLOv26s -- FULL AUGMENTATION
# Loads best HP config from: best_sweep_config_yolo26s_full.json
# ============================================================

set -e

WANDB_API_KEY="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"

cd ~/code/iToBoS
source ~/code/iToBoS/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY="${WANDB_API_KEY}"

echo "=============================================="
echo "  iToBoS | YOLOv26s Full Training -- FULL AUG"
echo "  Output: yamin/experiment-v26/runs/best_model/"
echo "=============================================="

python yamin/experiment-v26/train_best.py --aug_mode full --device 1

echo ""
echo "  ✓ Done! Weights at:"
echo "    yamin/experiment-v26/runs/best_model/yolo26s_best_full_100ep/weights/best.pt"
echo "  Next step: bash yamin/experiment-v26/run_evaluate_full.sh"
