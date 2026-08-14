#!/usr/bin/env bash
# ============================================================
# run_10_saliency_matched.sh
#
# Hayat's fix: Generate EigenCAM maps for BOTH YOLOv12s and
# D-FINE-S on the SAME set of images so they can be directly
# compared in the paper.
#
# Problem: run_09 ran YOLO on images 0070,0074,... and D-FINE
#          on images 0009,0018,... (different images because each
#          model picks the first N images where IT detects
#          something). This script forces both models to use
#          the SAME shared images.
#
# Output: results/saliency_matched/
#   eigencam_yolo/   – YOLOv12s EigenCAM panels
#   eigencam_dfine/  – D-FINE-S EigenCAM panels
#   combined/        – side-by-side comparison panels (SEND THESE TO HAYAT)
#
# Runtime: ~5-15 min for 6 images
# ============================================================
set -euo pipefail
source ~/code/iToBoS/aritra/dfine_code/dfine_env/bin/activate

EXPERIMENT_DIR=~/code/iToBoS/yamin/experiment-dfine

DFINE_CONFIG="$EXPERIMENT_DIR/results/configs/dfine_s_fullaug_sweep_100ep.yml"
DFINE_WEIGHTS="$EXPERIMENT_DIR/results/training/full/dfine_s_fullaug_sweep_100ep/best_stg2.pth"
DFINE_REPO=~/code/iToBoS/aritra/dfine_code/D-FINE
DATASET_DIR=~/code/iToBoS/aritra/dfine_results/dataset_coco

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
    echo "[ERROR] YOLO weights not found. Set YOLO_WEIGHTS manually."
    exit 1
fi

OUTPUT_DIR="$EXPERIMENT_DIR/results/saliency_matched"
mkdir -p "$OUTPUT_DIR"

echo "=== Matched EigenCAM Generation (YOLO + D-FINE on same images) ==="

# Install dependencies if needed
pip install grad-cam ultralytics --quiet 2>/dev/null || true

# ── Option A: Auto-select 6 images where BOTH models detect ──────────────────
python "$EXPERIMENT_DIR/scripts/saliency_matched.py" \
  --yolo-weights  "$YOLO_WEIGHTS" \
  --dfine-config  "$DFINE_CONFIG" \
  --dfine-weights "$DFINE_WEIGHTS" \
  --dfine-repo    "$DFINE_REPO" \
  --dataset-dir   "$DATASET_DIR" \
  --split test \
  --n-images 6 \
  --min-conf 0.25 \
  --device cuda:3 \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$OUTPUT_DIR/run.log"

# ── Option B: Use specific image IDs (uncomment to force images from paper) ───
# Replace the stems below with whatever images Hayat wants:
#
# python "$EXPERIMENT_DIR/scripts/saliency_matched.py" \
#   --yolo-weights  "$YOLO_WEIGHTS" \
#   --dfine-config  "$DFINE_CONFIG" \
#   --dfine-weights "$DFINE_WEIGHTS" \
#   --dfine-repo    "$DFINE_REPO" \
#   --dataset-dir   "$DATASET_DIR" \
#   --split test \
#   --n-images 6 \
#   --min-conf 0.20 \
#   --device cuda:3 \
#   --output-dir "$OUTPUT_DIR" \
#   --image-ids "test_000017_image_0070" "test_000001_image_0009" \
#   2>&1 | tee "$OUTPUT_DIR/run_pinned.log"

echo ""
echo "=== Done ==="
echo "Results: $OUTPUT_DIR"
echo ""
echo "Send Hayat the combined/ subfolder:"
echo "  $OUTPUT_DIR/combined/"
echo ""
echo "To zip and send:"
echo "  zip -r saliency_matched_combined.zip $OUTPUT_DIR/combined/"
