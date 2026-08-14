#!/usr/bin/env bash
# ============================================================
# Step 4 — Threshold sweep + PR curve for trained models
#
# Evaluates: full-aug model, no-aug model, and all ablations
# on the TEST set. Produces:
#   - threshold_sweep_test.csv  (11x9 conf x NMS-IoU grid)
#   - pr_curve_test.png
#   - threshold_sweep_f1_heatmap_test.png
#   - best_threshold_summary.json
#   - detections_best_test/    (25 annotated images)
#
# Run in tmux: tmux new -s dfine_eval
# ============================================================
set -euo pipefail
source ~/code/iToBoS/aritra/dfine_code/dfine_env/bin/activate

EXPERIMENT_DIR=~/code/iToBoS/yamin/experiment-dfine
DFINE_REPO=~/code/iToBoS/aritra/dfine_code/D-FINE
ARITRA_SCRIPTS=~/code/iToBoS/aritra/dfine_code/itobos_dfine_code/scripts
DATASET=~/code/iToBoS/aritra/dfine_results/dataset_coco
EVAL_OUT="$EXPERIMENT_DIR/results/evaluation"

# ── Helper function ────────────────────────────────────────────────────────
evaluate_model() {
  local label="$1"
  local run_name="$2"
  local aug_or_ablation="$3"   # "full", "noaug", or "ablations"

  # Try to find the output directory, handling optional _v2 suffix
  local out_dir=""
  for candidate_dir in "$EXPERIMENT_DIR/results/training/$aug_or_ablation/$run_name" \
                       "$EXPERIMENT_DIR/results/training/$aug_or_ablation/${run_name}_v2"; do
    if [ -d "$candidate_dir" ]; then
      out_dir="$candidate_dir"
      break
    fi
  done

  if [ -z "$out_dir" ]; then
    echo "[skip] No training directory found for $label (checked $run_name and ${run_name}_v2)"
    return
  fi

  # Config path selection
  local config=""
  if [ "$aug_or_ablation" = "ablations" ]; then
    # Ablations use the same config as fullaug_sweep but were trained with different loss weights
    config="$EXPERIMENT_DIR/results/configs/dfine_s_fullaug_sweep_100ep.yml"
  else
    config="$EXPERIMENT_DIR/results/configs/${run_name}.yml"
  fi

  # Prefer stg2 checkpoint, fall back to stg1.
  local weights="$out_dir/best_stg2.pth"
  [ -f "$weights" ] || weights="$out_dir/best_stg1.pth"

  if [ ! -f "$weights" ]; then
    echo "[skip] No checkpoint found for $label at $weights"
    return
  fi

  echo ""
  echo "================================================================"
  echo "  Evaluating: $label"
  echo "================================================================"

  python "$ARITRA_SCRIPTS/dfine_infer_eval.py" \
    --dfine-repo "$DFINE_REPO" \
    --config "$config" \
    --weights "$weights" \
    --dataset-dir "$DATASET" \
    --split test \
    --img-size 1024 \
    --device cuda:0 \
    --output-dir "$EVAL_OUT/$label" \
    --save-vis 25

  echo "  Done: $EVAL_OUT/$label"
}

# ── Evaluate main models ───────────────────────────────────────────────────
# (Commented out because they already completed successfully)
# evaluate_model "fullaug_sweep"   "dfine_s_fullaug_sweep_100ep"   "full"
# evaluate_model "noaug_sweep"     "dfine_s_noaug_sweep_100ep"     "noaug"

# ── Evaluate ablations ─────────────────────────────────────────────────────
evaluate_model "ablation_fgl0"      "ablation_fgl0_50ep"      "ablations"
evaluate_model "ablation_fgl_high"  "ablation_fgl_high_50ep"  "ablations"
evaluate_model "ablation_ddf0"      "ablation_ddf0_50ep"      "ablations"

echo ""
echo "=== All evaluations complete ==="
echo "Results in: $EVAL_OUT/"
