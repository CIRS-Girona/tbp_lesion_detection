#!/bin/bash
# ==============================================================
# run_quick_train.sh
# Quick 20-epoch RT-DETR-L training on iToBoS — feasibility check
# Run from: ~/code/iToBoS/
# ==============================================================

set -e

WANDB_API_KEY="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"

cd ~/code/iToBoS
source ~/code/iToBoS/.venv/bin/activate
export WANDB_API_KEY="${WANDB_API_KEY}"

echo "=============================================="
echo "  iToBoS | RT-DETR-L Quick Training (20 ep)"
echo "  Feasibility check — decide by Day 7"
echo "=============================================="

# Check GPU before starting — pick a free one
echo ""
echo "Current GPU status:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.free \
           --format=csv,noheader
echo ""

# Run on GPU 0 by default (change --device if GPU 0 is busy)
CUDA_VISIBLE_DEVICES=0 python praveen/experiment-rtdetr/quick_train.py \
    --epochs 20 \
    --device 0 \
    --batch 8 \
    --patience 10

echo ""
echo "  Done! Check WandB for results:"
echo "  https://wandb.ai/myamin-cs-universitat-de-girona/skin-lesion-detection"
echo ""
echo "  Weights saved to:"
echo "    praveen/experiment-rtdetr/runs/quick_train/rtdetr-l_quick_20ep/weights/best.pt"
