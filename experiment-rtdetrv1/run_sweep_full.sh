#!/bin/bash
# ============================================================
# run_sweep_full.sh
# Bayesian HP sweep for RT-DETR-L -- FULL AUGMENTATION
# Usage: bash praveen/experiment-rtdetr/run_sweep_full.sh [N_TRIALS]
#   N_TRIALS defaults to 40
# Run from: ~/code/iToBoS/
# ============================================================

set -e

N_TRIALS="${1:-40}"
WANDB_API_KEY="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"

cd ~/code/iToBoS
source ~/code/iToBoS/.venv/bin/activate
export WANDB_API_KEY="${WANDB_API_KEY}"

echo "=============================================="
echo "  RT-DETR-L Bayesian Sweep -- FULL AUG"
echo "  Trials: ${N_TRIALS}"
echo "=============================================="

# Create the sweep and capture its ID
SWEEP_ID=$(wandb sweep --project skin-lesion-detection --entity myamin-cs-universitat-de-girona \
    praveen/experiment-rtdetr/sweep_config_rtdetr_full.yaml 2>&1 | \
    grep -oP 'wandb agent \K[^ ]+' | tail -1)

echo ""
echo "  Sweep ID: ${SWEEP_ID}"
echo "  Starting agent for ${N_TRIALS} trials..."
echo ""

wandb agent --count "${N_TRIALS}" "${SWEEP_ID}"

echo ""
echo "  Sweep done! Best config saved to:"
echo "    praveen/experiment-rtdetr/best_sweep_config_rtdetr-l_full.json"
echo "  Next: bash praveen/experiment-rtdetr/run_train_full.sh"
