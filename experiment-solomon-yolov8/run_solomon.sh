#!/usr/bin/env bash
# run_solomon.sh
# ──────────────────────────────────────────────────────────────────────────────
# Runs the complete Solomon baseline experiment:
#   1. Train full-aug (Solomon's original settings)
#   2. Evaluate full-aug  (default conf + 11x9 sweep)
#   3. Train no-aug       (ablation)
#   4. Evaluate no-aug    (default conf + 11x9 sweep)
#
# Run from ~/code/iToBoS/:
#   tmux new-session -d -s solomon -n train
#   tmux send-keys -t solomon:train \
#     "cd ~/code/iToBoS && source .venv/bin/activate && bash yamin/experiment-solomon/run_solomon.sh" Enter
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
fail() { echo "[ERROR] $*" >&2; exit 1; }

cd "$REPO_ROOT" || fail "Cannot cd to $REPO_ROOT"
log "Working directory: $(pwd)"

GPU=0

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Train full augmentation (Solomon's original setting)
# ──────────────────────────────────────────────────────────────────────────────
log "=== STEP 1: Training Solomon baseline — full augmentation ==="
CUDA_VISIBLE_DEVICES=$GPU python yamin/experiment-solomon/train.py \
    --aug_mode full \
    --device 0 \
    --epochs 100 \
    --patience 20

FULL_WEIGHTS="yamin/experiment-solomon/runs/best_model/solomon_yolov8s_full_100ep/weights/best.pt"
[[ -f "$FULL_WEIGHTS" ]] || fail "Full-aug weights not found at $FULL_WEIGHTS"
log "Full-aug weights: $FULL_WEIGHTS"

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Evaluate full augmentation
# ──────────────────────────────────────────────────────────────────────────────
log "=== STEP 2: Evaluating Solomon baseline — full augmentation ==="
CUDA_VISIBLE_DEVICES=$GPU python yamin/experiment-solomon/evaluate.py \
    --weights "$FULL_WEIGHTS" \
    --name    full \
    --device  0

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Train no augmentation (ablation)
# ──────────────────────────────────────────────────────────────────────────────
log "=== STEP 3: Training Solomon baseline — no augmentation ==="
CUDA_VISIBLE_DEVICES=$GPU python yamin/experiment-solomon/train.py \
    --aug_mode noaug \
    --device 0 \
    --epochs 100 \
    --patience 20

NOAUG_WEIGHTS="yamin/experiment-solomon/runs/best_model/solomon_yolov8s_noaug_100ep/weights/best.pt"
[[ -f "$NOAUG_WEIGHTS" ]] || fail "No-aug weights not found at $NOAUG_WEIGHTS"
log "No-aug weights: $NOAUG_WEIGHTS"

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — Evaluate no augmentation
# ──────────────────────────────────────────────────────────────────────────────
log "=== STEP 4: Evaluating Solomon baseline — no augmentation ==="
CUDA_VISIBLE_DEVICES=$GPU python yamin/experiment-solomon/evaluate.py \
    --weights "$NOAUG_WEIGHTS" \
    --name    noaug \
    --device  0

log "=== ALL DONE — Solomon baseline complete ==="
log "Results at: yamin/experiment-solomon/runs/"
log "WandB project: skin-lesion-detection (experiment-solomon tag)"
