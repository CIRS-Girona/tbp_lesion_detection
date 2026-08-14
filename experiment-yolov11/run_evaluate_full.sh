#!/bin/bash
# ============================================================
# run_evaluate_full.sh
# Threshold sweep evaluation for YOLOv11s -- FULL AUG model
# Location: yamin/experiment-v11/
#
# Run AFTER run_train_full.sh completes.
# Saves to: yamin/experiment-v11/runs/evaluation/full/
# Run from iToBoS root:
#   bash yamin/experiment-v11/run_evaluate_full.sh
# ============================================================

set -e

WANDB_API_KEY="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"
WEIGHTS="yamin/experiment-v11/runs/best_model/yolo11s_best_full_100ep/weights/best.pt"

cd ~/code/iToBoS
source ~/code/iToBoS/.venv/bin/activate
export WANDB_API_KEY="${WANDB_API_KEY}"

echo "=============================================="
echo "  iToBoS | YOLOv11s Evaluation -- FULL AUG"
echo "  11 conf × 9 NMS IoU = 99 combinations"
echo "  Saves to: runs/evaluation/full/"
echo "=============================================="

python yamin/experiment-v11/evaluate.py \
    --weights "${WEIGHTS}" \
    --run_name yolo11s_full_evaluation \
    --name full \
    --conf 0.20 \
    --iou 0.50

echo ""
echo "  ✓ Evaluation complete!"
echo "  Results: yamin/experiment-v11/runs/evaluation/full/"
