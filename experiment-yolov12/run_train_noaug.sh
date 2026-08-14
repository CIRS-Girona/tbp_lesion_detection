#!/bin/bash
# ============================================================
# run_train_noaug.sh
# Full 100-epoch training for YOLOv12s -- NO AUGMENTATION
# Loads best HP config from: best_sweep_config_yolo12s_noaug.json
# ============================================================

set -e

WANDB_API_KEY="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"

cd ~/code/iToBoS
source ~/code/iToBoS/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY="${WANDB_API_KEY}"

echo "=============================================="
echo "  iToBoS | YOLOv12s Full Training -- NO AUG"
echo "  Output: yamin/experiment-v12/runs/best_model/"
echo "=============================================="

python yamin/experiment-v12/train_best.py --aug_mode noaug

echo ""
echo "  ✓ Done! Weights at:"
echo "    yamin/experiment-v12/runs/best_model/yolo12s_best_noaug_100ep/weights/best.pt"
echo "  Next step: bash yamin/experiment-v12/run_evaluate_noaug.sh"
