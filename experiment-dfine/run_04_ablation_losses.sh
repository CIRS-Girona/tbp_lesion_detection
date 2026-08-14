#!/usr/bin/env bash
# ============================================================
# Step 3 — Loss function ablation study (for paper analysis)
#
# Three variants × 50 epochs each:
#   A) FGL weight = 0    (removes D-FINE's key innovation)
#   B) FGL weight = 0.30 (2× the default 0.15)
#   C) DDF weight = 0    (removes inter-layer distillation)
#
# These quantify how much each D-FINE-specific loss contributes
# to the final mAP50. Run sequentially in one tmux session.
# Expected runtime: ~3 × 3.5h = ~10.5 hours.
# Run in tmux: tmux new -s dfine_ablation
# ============================================================
set -euo pipefail
source ~/code/iToBoS/aritra/dfine_code/dfine_env/bin/activate

EXPERIMENT_DIR=~/code/iToBoS/yamin/experiment-dfine
DFINE_REPO=~/code/iToBoS/aritra/dfine_code/D-FINE
MAKE_CFG="$EXPERIMENT_DIR/scripts/make_dfine_config_yamin.py"

# Best sweep HPs from trial 11
BEST_LR=1.2792e-4
BEST_BB_LR=1.2792e-4
BEST_WD=4.0198e-5
ABLATION_EPOCHS=50

run_ablation() {
  local gpu="$1"
  local run_name="$2"
  shift 2
  # Remaining args are loss weight overrides e.g. --loss-fgl 0.0

  echo ""
  echo "================================================================"
  echo "  Ablation: $run_name  (GPU $gpu)"
  echo "================================================================"

  # Re-use the working fullaug config — wrapper patches weight_dict in memory.
  # This avoids D-FINE's YAML criterion format issues entirely.
  BASE_CONFIG="$EXPERIMENT_DIR/results/configs/dfine_s_fullaug_sweep_100ep.yml"
  OUTPUT="$EXPERIMENT_DIR/results/training/ablations/$run_name"
  mkdir -p "$OUTPUT"

  CUDA_VISIBLE_DEVICES=$gpu python \
    "$EXPERIMENT_DIR/scripts/train_ablation_dfine.py" \
    --config "$BASE_CONFIG" \
    --output-dir "$OUTPUT" \
    --seed 42 \
    --use-amp \
    "$@" \
    2>&1 | tee "$OUTPUT/train_stdout.log"

  echo "  Done: $run_name"
}

# NOTE: Run each ablation in a SEPARATE tmux session on a different GPU!
# See commands below — do NOT run this script directly.
#
# tmux new -s abl_fgl0  → bash run_04_ablation_losses.sh fgl0
# tmux new -s abl_ddf0  → bash run_04_ablation_losses.sh ddf0
# tmux new -s abl_fglhi → bash run_04_ablation_losses.sh fgl_high

TARGET="${1:-all}"   # pass 'fgl0', 'ddf0', 'fgl_high', or 'all'

case "$TARGET" in
  fgl0)
    # Ablation A: FGL = 0 (removes D-FINE's distribution refinement loss)
    run_ablation 1 "ablation_fgl0_${ABLATION_EPOCHS}ep" --loss-fgl 0.0
    ;;
  fgl_high)
    # Ablation B: FGL = 0.30 (2x default of 0.15)
    run_ablation 2 "ablation_fgl_high_${ABLATION_EPOCHS}ep" --loss-fgl 0.30
    ;;
  ddf0)
    # Ablation C: DDF = 0 (removes inter-layer distillation)
    run_ablation 3 "ablation_ddf0_${ABLATION_EPOCHS}ep" --loss-ddf 0.0
    ;;
  all)
    # Sequential fallback — use separate tmux sessions instead!
    run_ablation 0 "ablation_fgl0_${ABLATION_EPOCHS}ep"     --loss-fgl 0.0
    run_ablation 0 "ablation_fgl_high_${ABLATION_EPOCHS}ep" --loss-fgl 0.30
    run_ablation 0 "ablation_ddf0_${ABLATION_EPOCHS}ep"     --loss-ddf 0.0
    ;;
esac

echo ""
echo "=== Ablation done: $TARGET ==="
