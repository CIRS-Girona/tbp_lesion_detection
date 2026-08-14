#!/bin/bash
# ============================================================
# run_evaluate_noaug.sh
# Threshold sweep evaluation for YOLOv26s -- NO AUGMENTATION
# ============================================================

set -e

WANDB_API_KEY="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"
WEIGHTS="yamin/experiment-v26/runs/best_model/yolo26s_best_noaug_100ep/weights/best.pt"

cd ~/code/iToBoS
source ~/code/iToBoS/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY="${WANDB_API_KEY}"

echo "=============================================="
echo "  iToBoS | YOLOv26s Evaluation -- NO AUG"
echo "  Weights: ${WEIGHTS}"
echo "=============================================="

python yamin/experiment-v26/evaluate.py \
    --weights "${WEIGHTS}" \
    --run_name "yolo26s_noaug_evaluation" \
    --name "noaug" \
    --conf 0.20 \
    --iou 0.50 \
    --device 0

echo ""
echo "  ✓ Results saved to: yamin/experiment-v26/runs/evaluation/"
