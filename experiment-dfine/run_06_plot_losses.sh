#!/usr/bin/env bash
# ============================================================
# Step 5 — Plot loss curves + gradient flow analysis
#
# For each trained model, plots:
#   - training_loss_curves_all.png (all losses together)
#   - training_loss_fgl.png        (FGL only)
#   - training_loss_ddf.png        (DDF only)
#   - training_loss_vfl.png        (VFL only)
#   - training_loss_giou.png       (GIoU only)
#   - loss_curves.csv
#   - gradient_flow/ PNGs/CSVs (from Step 0 patch)
#
# Run interactively (fast, minutes not hours).
# ============================================================
set -euo pipefail
source ~/code/iToBoS/aritra/dfine_code/dfine_env/bin/activate

EXPERIMENT_DIR=~/code/iToBoS/yamin/experiment-dfine
ARITRA_SCRIPTS=~/code/iToBoS/aritra/dfine_code/itobos_dfine_code/scripts

plot_losses() {
  local label="$1"
  local run_name="$2"
  local aug_or_ablation="$3"

  local log="$EXPERIMENT_DIR/results/training/$aug_or_ablation/$run_name/log.txt"
  local out="$EXPERIMENT_DIR/results/training/$aug_or_ablation/$run_name/plots"

  if [ ! -f "$log" ]; then
    echo "[skip] No log found: $log"
    return
  fi
  echo "Plotting: $label"
  python "$ARITRA_SCRIPTS/plot_losses.py" \
    --log "$log" \
    --out-dir "$out"
}

# Main trained models
plot_losses "fullaug_sweep"  "dfine_s_fullaug_sweep_100ep"  "full"
plot_losses "noaug_sweep"    "dfine_s_noaug_sweep_100ep"    "noaug"

# Ablations (50-epoch runs)
plot_losses "ablation_fgl0"      "ablation_fgl0_50ep"      "ablations"
plot_losses "ablation_fgl_high"  "ablation_fgl_high_50ep"  "ablations"
plot_losses "ablation_ddf0"      "ablation_ddf0_50ep"      "ablations"

echo ""
echo "=== Loss plots saved in each model's plots/ subdirectory ==="
echo ""
echo "For gradient flow PNGs, look in:"
echo "  <output_dir>/gradient_flow/grad_flow_epochXXXX_stepXXXXXX.png"
