#!/usr/bin/env bash
# run_msdetr.sh
# ──────────────────────────────────────────────────────────────────────────────
# Complete MS-DETR experiment pipeline:
#   1. Create dedicated Python venv + install patched ultralytics (once)
#   2. Sweep full aug  (15 trials x 30 epochs)
#   3. Train full aug  (100 epochs, best HPs)
#   4. Evaluate full aug (default conf + 11x9 sweep)
#   5. Sweep no aug    (15 trials x 30 epochs)
#   6. Train no aug    (100 epochs, best HPs)
#   7. Evaluate no aug (default conf + 11x9 sweep)
#
# NOTE: Uses a separate venv (msdetr-venv/) so the patched ultralytics
#       does NOT conflict with the existing .venv used for YOLO experiments.
#
# Run from ~/code/iToBoS/:
#   tmux new-session -d -s msdetr -n main
#   tmux send-keys -t msdetr:main \
#     "cd ~/code/iToBoS && bash yamin/experiment-msdetr/run_msdetr.sh" Enter
#   tmux attach -t msdetr
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
MSDETR_SRC="$REPO_ROOT/yamin/msdetr-src"
VENV_DIR="$REPO_ROOT/yamin/msdetr-venv"

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
fail() { echo "[ERROR] $*" >&2; exit 1; }

cd "$REPO_ROOT" || fail "Cannot cd to $REPO_ROOT"
log "Working directory: $(pwd)"

# ──────────────────────────────────────────────────────────────────────────────
# STEP 0 — Create dedicated Python venv for MS-DETR (idempotent)
# ──────────────────────────────────────────────────────────────────────────────
log "=== STEP 0: Setting up MS-DETR Python venv ==="

[[ -d "$MSDETR_SRC" ]] || fail "MSDETR source not found at $MSDETR_SRC"

# Create venv if it doesn't already exist
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    log "Creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

# Activate the MSDETR venv (replaces .venv if it was active)
source "$VENV_DIR/bin/activate"
log "Active Python: $(which python3)  ($(python3 --version))"

# Install / upgrade pip
pip install --upgrade pip --quiet

# Install patched ultralytics (editable install — makes MDF/DSA/DFFB importable)
# Note: skipping requirements.txt — it has Python 3.12 pins that break on 3.10.
# We install only what MS-DETR actually needs, with flexible versions.
log "Installing patched ultralytics from $MSDETR_SRC (editable, no deps)"
pip install -e "$MSDETR_SRC" --no-deps --quiet

log "Installing MS-DETR runtime dependencies (flexible versions for Python 3.10)"
pip install \
    torch torchvision \
    wandb \
    einops \
    timm \
    thop \
    PyYAML \
    scipy \
    matplotlib \
    opencv-python \
    pillow \
    tqdm \
    prettytable \
    psutil \
    py-cpuinfo \
    seaborn \
    pandas \
    requests \
    dill \
    --quiet

log "Environment ready."

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Sweep full augmentation
# ──────────────────────────────────────────────────────────────────────────────
log "=== STEP 1: Hyperparameter sweep — full augmentation (15 trials x 30 epochs) ==="
CUDA_VISIBLE_DEVICES=1 python yamin/experiment-msdetr/hparam_sweep.py --aug_mode full
log "Sweep (full) complete."

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Train full aug (100 epochs)
# ──────────────────────────────────────────────────────────────────────────────
log "=== STEP 2: Training MS-DETR — full augmentation (100 epochs) ==="
CUDA_VISIBLE_DEVICES=1 python yamin/experiment-msdetr/train_best.py --aug_mode full
FULL_WEIGHTS="yamin/experiment-msdetr/runs/best_model/msdetr_best_full_100ep/weights/best.pt"
[[ -f "$FULL_WEIGHTS" ]] || fail "Full-aug weights not found: $FULL_WEIGHTS"

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Evaluate full aug
# ──────────────────────────────────────────────────────────────────────────────
log "=== STEP 3: Evaluating MS-DETR — full augmentation ==="
CUDA_VISIBLE_DEVICES=1 python yamin/experiment-msdetr/evaluate.py \
    --weights "$FULL_WEIGHTS" \
    --name    full

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — Sweep no augmentation
# ──────────────────────────────────────────────────────────────────────────────
log "=== STEP 4: Hyperparameter sweep — no augmentation (15 trials x 30 epochs) ==="
CUDA_VISIBLE_DEVICES=1 python yamin/experiment-msdetr/hparam_sweep.py --aug_mode noaug
log "Sweep (noaug) complete."

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — Train no aug (100 epochs)
# ──────────────────────────────────────────────────────────────────────────────
log "=== STEP 5: Training MS-DETR — no augmentation (100 epochs) ==="
CUDA_VISIBLE_DEVICES=1 python yamin/experiment-msdetr/train_best.py --aug_mode noaug
NOAUG_WEIGHTS="yamin/experiment-msdetr/runs/best_model/msdetr_best_noaug_100ep/weights/best.pt"
[[ -f "$NOAUG_WEIGHTS" ]] || fail "No-aug weights not found: $NOAUG_WEIGHTS"

# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — Evaluate no aug
# ──────────────────────────────────────────────────────────────────────────────
log "=== STEP 6: Evaluating MS-DETR — no augmentation ==="
CUDA_VISIBLE_DEVICES=1 python yamin/experiment-msdetr/evaluate.py \
    --weights "$NOAUG_WEIGHTS" \
    --name    noaug

log "=== ALL DONE — MS-DETR experiment complete ==="
log "Results: yamin/experiment-msdetr/runs/"
log "WandB: skin-lesion-detection project (experiment-msdetr tag)"
