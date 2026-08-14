#!/usr/bin/env bash
# ============================================================
# Step 2b — Full 100-epoch training, NO AUGMENTATION
#
# Use the same best LR/WD from the sweep.
# Expected runtime: ~5-6 hours on A100.
# Run in tmux: tmux new -s dfine_noaug
# ============================================================
set -euo pipefail
source ~/code/iToBoS/aritra/dfine_code/dfine_env/bin/activate

EXPERIMENT_DIR=~/code/iToBoS/yamin/experiment-dfine
DFINE_REPO=~/code/iToBoS/aritra/dfine_code/D-FINE

# ── Best HPs from sweep trial 11 (same as full-aug) ──────────────────
BEST_LR=1.2792e-4
BEST_BB_LR=1.2792e-4    # backbone_lr_ratio=1.0 (sweep finding)
BEST_WD=4.0198e-5
RUN_NAME="dfine_s_noaug_sweep_100ep"

echo "=== D-FINE-S No-Aug Training (100 epochs) ==="
echo "lr=$BEST_LR  backbone_lr=$BEST_BB_LR  wd=$BEST_WD"

python "$EXPERIMENT_DIR/scripts/make_dfine_config_yamin.py" \
  --aug-mode noaug \
  --epochs 100 \
  --lr "$BEST_LR" \
  --backbone-lr "$BEST_BB_LR" \
  --weight-decay "$BEST_WD" \
  --run-name "$RUN_NAME"

CONFIG="$EXPERIMENT_DIR/results/configs/${RUN_NAME}.yml"
OUTPUT="$EXPERIMENT_DIR/results/training/noaug/$RUN_NAME"

CUDA_VISIBLE_DEVICES=2 python "$DFINE_REPO/train.py" \
  -c "$CONFIG" \
  --seed 42 \
  --use-amp \
  --output-dir "$OUTPUT" \
  2>&1 | tee "$OUTPUT/train_stdout.log"

echo ""
echo "=== Training complete ==="
echo "Best checkpoint: $OUTPUT/best_stg2.pth (or best_stg1.pth)"
