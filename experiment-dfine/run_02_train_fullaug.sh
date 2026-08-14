#!/usr/bin/env bash
# ============================================================
# Step 2a — Full 100-epoch training, FULL AUGMENTATION
#
# Edit BEST_LR, BEST_BB_LR, BEST_WD with the best values from
# the WandB sweep (or use Aritra's values as starting point).
#
# Aritra's values (fixed HPs, no sweep):
#   lr=5e-5, backbone_lr=2.5e-5, weight_decay=1e-4
#
# Expected runtime: ~6-7 hours on A100.
# Run in tmux: tmux new -s dfine_fullaug
# ============================================================
set -euo pipefail
source ~/code/iToBoS/aritra/dfine_code/dfine_env/bin/activate

EXPERIMENT_DIR=~/code/iToBoS/yamin/experiment-dfine
DFINE_REPO=~/code/iToBoS/aritra/dfine_code/D-FINE

# ── Best HPs from sweep trial 11 (val_mAP50=0.641 @30ep) ────────────────
BEST_LR=1.2792e-4
BEST_BB_LR=1.2792e-4    # backbone_lr_ratio=1.0 → same as lr (sweep finding)
BEST_WD=4.0198e-5
RUN_NAME="dfine_s_fullaug_sweep_100ep"

echo "=== D-FINE-S Full Aug Training (100 epochs) ==="
echo "lr=$BEST_LR  backbone_lr=$BEST_BB_LR  wd=$BEST_WD"

# Generate config
python "$EXPERIMENT_DIR/scripts/make_dfine_config_yamin.py" \
  --aug-mode full \
  --epochs 100 \
  --lr "$BEST_LR" \
  --backbone-lr "$BEST_BB_LR" \
  --weight-decay "$BEST_WD" \
  --run-name "$RUN_NAME"

CONFIG="$EXPERIMENT_DIR/results/configs/${RUN_NAME}.yml"
OUTPUT="$EXPERIMENT_DIR/results/training/full/$RUN_NAME"

CUDA_VISIBLE_DEVICES=0 python "$DFINE_REPO/train.py" \
  -c "$CONFIG" \
  --seed 42 \
  --use-amp \
  --output-dir "$OUTPUT" \
  2>&1 | tee "$OUTPUT/train_stdout.log"

echo ""
echo "=== Training complete ==="
echo "Best checkpoint: $OUTPUT/best_stg2.pth (or best_stg1.pth)"
