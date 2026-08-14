#!/bin/bash
# ============================================================
# run_train_noaug.sh
# Full 100-epoch training for YOLOv8s -- NO AUGMENTATION
# Location: yamin/experiment-v8/
#
# Run AFTER run_sweep_noaug.sh completes.
# Run from iToBoS root:
#   bash yamin/experiment-v8/run_train_noaug.sh
# ============================================================

set -e

WANDB_API_KEY="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"

cd ~/code/iToBoS
source ~/code/iToBoS/.venv/bin/activate
export WANDB_API_KEY="${WANDB_API_KEY}"

echo "=============================================="
echo "  iToBoS | YOLOv8s Full Training -- NO AUG"
echo "  100 epochs, patience=20"
echo "=============================================="

python yamin/experiment-v8/train_best.py \
    --aug_mode noaug \
    --epochs 100 \
    --patience 20

echo ""
echo "  Done! Weights at:"
echo "    yamin/experiment-v8/runs/best_model/yolov8s_best_noaug_100ep/weights/best.pt"
echo "  Next step: bash yamin/experiment-v8/run_evaluate_noaug.sh"
