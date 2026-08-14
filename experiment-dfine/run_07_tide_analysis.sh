#!/usr/bin/env bash
# ============================================================
# Step 6 — TIDE Error Analysis + NMS vs NMS-free analysis
#
# Requires: pip install tidecv
# Requires: COCO-format prediction JSONs for each model.
#
# D-FINE preds are auto-saved by run_05_evaluate_all.sh at:
#   results/evaluation/<label>/dfine_raw_preds.json
#
# YOLO predictions need to be saved separately — see note below.
# Run in a terminal (fast, minutes): tmux new -s tide
# ============================================================
set -euo pipefail
source ~/code/iToBoS/aritra/dfine_code/dfine_env/bin/activate

EXPERIMENT_DIR=~/code/iToBoS/yamin/experiment-dfine
EVAL_DIR="$EXPERIMENT_DIR/results/evaluation"
GT_JSON=~/code/iToBoS/aritra/dfine_results/dataset_coco/annotations/instances_test.json

# Install TIDE if not already
pip install tidecv --quiet 2>/dev/null || true

echo "=== TIDE Error Analysis ==="

# NOTE: YOLO prediction JSONs must be saved in COCO format.
# If you ran YOLO evaluation with Ultralytics, save predictions with:
#   model.val(data=..., save_json=True)  → runs/detect/val/predictions.json
# Place them at the paths below. Skip models you don't have predictions for.

# Direct path to YOLOv12 prediction JSON
YOLO12_PREDS=~/code/iToBoS/yamin/experiment-v12/runs/evaluation/full/test_conf0.20_iou0.50/predictions.json

YOLO_ARGS=""
if [ -f "$YOLO12_PREDS" ]; then
    YOLO_ARGS="--yolov12-preds $YOLO12_PREDS"
    echo "[found] YOLOv12 prediction JSON: $YOLO12_PREDS"
else
    echo "[warn] YOLOv12 prediction JSON not found at $YOLO12_PREDS"
fi

# Create output dir BEFORE running the python pipeline so tee doesn't fail
mkdir -p "$EVAL_DIR/tide_analysis"

python "$EXPERIMENT_DIR/scripts/tide_analysis.py" \
  --split test \
  --gt-json "$GT_JSON" \
  --output-dir "$EVAL_DIR/tide_analysis" \
  --dfine-fullaug "$EVAL_DIR/fullaug_sweep/test/raw_predictions_conf0001.json" \
  --dfine-noaug   "$EVAL_DIR/noaug_sweep/test/raw_predictions_conf0001.json" \
  $YOLO_ARGS \
  2>&1 | tee "$EVAL_DIR/tide_analysis/tide_run.log"

echo ""
echo "=== TIDE analysis complete ==="
echo "Results: $EVAL_DIR/tide_analysis/"
