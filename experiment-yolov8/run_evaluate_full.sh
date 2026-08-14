#!/bin/bash
# ============================================================
# run_evaluate_full.sh
# Threshold sweep evaluation for YOLOv8s -- FULL AUGMENTATION
# Location: yamin/experiment-v8/
#
# Run AFTER run_train_full.sh completes.
# Run from iToBoS root:
#   bash yamin/experiment-v8/run_evaluate_full.sh
# ============================================================

set -e

WANDB_API_KEY="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"

cd ~/code/iToBoS
source ~/code/iToBoS/.venv/bin/activate
export WANDB_API_KEY="${WANDB_API_KEY}"

echo "=============================================="
echo "  iToBoS | YOLOv8s Evaluation -- FULL AUG"
echo "  11 x 9 conf x NMS IoU threshold sweep"
echo "=============================================="

python yamin/experiment-v8/evaluate.py \
    --weights yamin/experiment-v8/runs/best_model/yolov8s_best_full_100ep/weights/best.pt \
    --run_name yolov8s_full_evaluation \
    --name full \
    --conf 0.20 \
    --iou 0.50

echo ""
echo "  Evaluation complete."
echo "  Results at: yamin/experiment-v8/runs/evaluation/full/"
