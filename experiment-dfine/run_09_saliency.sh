#!/usr/bin/env bash
# ============================================================
# Step 9 — Grad-CAM/EigenCAM + D-FINE Cross-Attention Saliency
#
# Implements Hayat's "Clever Hans" check:
#   - Does the model focus on the lesion or background?
#   - EigenCAM on YOLO detection head
#   - Cross-attention maps from D-FINE decoder
#
# Requires: pip install grad-cam
# Runtime: ~15-30 min for 12 images
# ============================================================
set -euo pipefail
source ~/code/iToBoS/aritra/dfine_code/dfine_env/bin/activate

EXPERIMENT_DIR=~/code/iToBoS/yamin/experiment-dfine

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
    echo "[warn] YOLO weights not found — EigenCAM will be skipped for YOLO."
    echo "       Set YOLO_WEIGHTS manually if needed."
fi

# Install grad-cam and ultralytics if needed
pip install grad-cam ultralytics --quiet 2>/dev/null || true

echo "=== Saliency / Attention Analysis (Clever Hans check) ==="

# Create output dir BEFORE tee tries to write the log (this was the first error)
mkdir -p "$EXPERIMENT_DIR/results/saliency"

# Build python args — only pass --yolo-weights if we found the file
if [ -n "$YOLO_WEIGHTS" ]; then
    MODE="both"
    YOLO_ARG="--yolo-weights $YOLO_WEIGHTS"
else
    MODE="dfine_attn"
    YOLO_ARG=""
fi

python "$EXPERIMENT_DIR/scripts/saliency_analysis.py" \
  --mode "$MODE" \
  --dfine-config  "$DFINE_CONFIG" \
  --dfine-weights "$DFINE_WEIGHTS" \
  $YOLO_ARG \
  --split test \
  --n-images 12 \
  --min-conf 0.25 \
  --device cuda:3 \
  --output-dir "$EXPERIMENT_DIR/results/saliency" \
  2>&1 | tee "$EXPERIMENT_DIR/results/saliency/run.log"

echo ""
echo "=== Saliency analysis complete ==="
echo "Results: $EXPERIMENT_DIR/results/saliency/"
echo ""
echo "Paper caption guidance:"
echo "  EigenCAM: 'YOLOv12s detection head activations — model attends to'"
echo "            'lesion centre (correct focus, no Clever Hans effect)'"
echo "  D-FINE:   'D-FINE decoder query cross-attention — peaked spatial'"
echo "            'attention confirms object-centric localisation'"
