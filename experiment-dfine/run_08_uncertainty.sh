#!/usr/bin/env bash
# ============================================================
# Step 7 — Localization Uncertainty Distribution Analysis
#
# Plots:
#   - D-FINE coordinate probability distributions (peaked = confident)
#   - YOLO DFL distributions
#   - Entropy comparison D-FINE vs YOLO
#   - Confidence vs entropy scatter
#
# Run AFTER training and evaluation are done.
# Fast (minutes): tmux new -s uncertainty
# ============================================================
set -euo pipefail
source ~/code/iToBoS/aritra/dfine_code/dfine_env/bin/activate

# Install timm and einops if needed (dependencies of YOLOv12 architecture)
pip install timm einops --quiet 2>/dev/null || true

EXPERIMENT_DIR=~/code/iToBoS/yamin/experiment-dfine

# Best trained D-FINE model
DFINE_CONFIG="$EXPERIMENT_DIR/results/configs/dfine_s_fullaug_sweep_100ep.yml"
DFINE_WEIGHTS="$EXPERIMENT_DIR/results/training/full/dfine_s_fullaug_sweep_100ep/best_stg2.pth"

# Auto-discover YOLO best.pt — search common locations
YOLO_WEIGHTS=""
for candidate in \
    ~/code/iToBoS/yamin/experiment-v12/runs/best_model/yolo12s_best_full_100ep/weights/best.pt \
    ~/code/iToBoS/yamin/experiment-yolo/yolov12s_best.pt \
    ~/code/iToBoS/yamin/experiment-yolo/runs/detect/yolov12s_fullaug/weights/best.pt \
    ~/code/iToBoS/yamin/experiment-yolo/runs/detect/yolov12s/weights/best.pt \
    ~/code/iToBoS/aritra/dfine_results/yolov12s_best.pt \
    ~/code/iToBoS/yamin/experiment-yolo/best.pt; do
    expanded=$(eval echo "$candidate")
    if [ -f "$expanded" ]; then
        YOLO_WEIGHTS="$expanded"
        echo "[found] YOLO weights: $YOLO_WEIGHTS"
        break
    fi
done
if [ -z "$YOLO_WEIGHTS" ]; then
    echo "[warn] YOLO weights not found — uncertainty analysis for YOLO will be skipped."
fi

echo "=== Uncertainty Distribution Analysis ==="

# Create output dir BEFORE running the python pipeline so tee doesn't fail
mkdir -p "$EXPERIMENT_DIR/results/uncertainty_analysis"

python "$EXPERIMENT_DIR/scripts/uncertainty_distribution.py" \
  --split test \
  --dfine-config  "$DFINE_CONFIG" \
  --dfine-weights "$DFINE_WEIGHTS" \
  --n-images 25 \
  --min-conf 0.3 \
  --output-dir "$EXPERIMENT_DIR/results/uncertainty_analysis" \
  2>&1 | tee "$EXPERIMENT_DIR/results/uncertainty_analysis/run.log"

# If YOLO weights exist, also run with YOLO
if [ -f "$YOLO_WEIGHTS" ]; then
  python "$EXPERIMENT_DIR/scripts/uncertainty_distribution.py" \
    --split test \
    --dfine-config  "$DFINE_CONFIG" \
    --dfine-weights "$DFINE_WEIGHTS" \
    --yolo-weights  "$YOLO_WEIGHTS" \
    --n-images 25 \
    --min-conf 0.3 \
    --output-dir "$EXPERIMENT_DIR/results/uncertainty_analysis" \
    2>&1 | tee -a "$EXPERIMENT_DIR/results/uncertainty_analysis/run.log"
fi

echo ""
echo "=== Uncertainty analysis complete ==="
echo "Results: $EXPERIMENT_DIR/results/uncertainty_analysis/"
