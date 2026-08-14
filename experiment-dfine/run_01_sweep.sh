#!/usr/bin/env bash
# ============================================================
# Step 1 — Bayesian hyperparameter sweep (20 trials × 30 epochs)
# Expected runtime: ~12 hours on A100. Run in tmux.
# ============================================================
set -euo pipefail
source ~/code/iToBoS/aritra/dfine_code/dfine_env/bin/activate

cd ~/code/iToBoS/yamin/experiment-dfine

echo "=== Creating WandB sweep ==="
SWEEP_OUTPUT=$(wandb sweep sweep_config.yaml 2>&1)
echo "$SWEEP_OUTPUT"

# Extract the sweep ID from output line: "wandb agent entity/project/id"
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP '(?<=wandb agent )\S+' | tail -1)

if [ -z "$SWEEP_ID" ]; then
  echo "ERROR: Could not parse sweep ID from wandb output."
  echo "Run manually:  wandb sweep sweep_config.yaml  then  wandb agent <id>"
  exit 1
fi

echo ""
echo "=== Starting sweep agent (20 trials × 30 epochs) ==="
echo "Sweep ID: $SWEEP_ID"
wandb agent "$SWEEP_ID"
