#!/bin/bash
# ============================================================
# run_evaluate_noaug.sh
# Threshold sweep evaluation for YOLOv11s -- NO AUG model
# Location: yamin/experiment-v11/
#
# Run AFTER run_train_noaug.sh completes.
# Saves to: yamin/experiment-v11/runs/evaluation/noaug/
# Run from iToBoS root:
#   bash yamin/experiment-v11/run_evaluate_noaug.sh
# ============================================================

set -e

WANDB_API_KEY="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"
WEIGHTS="yamin/experiment-v11/runs/best_model/yolo11s_best_noaug_100ep/weights/best.pt"

cd ~/code/iToBoS
source ~/code/iToBoS/.venv/bin/activate
export WANDB_API_KEY="${WANDB_API_KEY}"

echo "=============================================="
echo "  iToBoS | YOLOv11s Evaluation -- NO AUG"
echo "  11 conf × 9 NMS IoU = 99 combinations"
echo "  Saves to: runs/evaluation/noaug/"
echo "=============================================="

python yamin/experiment-v11/evaluate.py \
    --weights "${WEIGHTS}" \
    --run_name yolo11s_noaug_evaluation \
    --name noaug \
    --conf 0.20 \
    --iou 0.50

echo ""
echo "  ✓ Evaluation complete!"
echo "  Results: yamin/experiment-v11/runs/evaluation/noaug/"
