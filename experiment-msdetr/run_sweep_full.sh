#!/usr/bin/env bash
# run_sweep_full.sh — MS-DETR sweep, full augmentation
# Run from ~/code/iToBoS/:
#   bash yamin/experiment-msdetr/run_sweep_full.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$REPO_ROOT/yamin/msdetr-venv/bin/activate"
[[ -f "$VENV" ]] || { echo "[ERROR] msdetr-venv not found. Run run_setup.sh first."; exit 1; }
source "$VENV"
cd "$REPO_ROOT"
echo "[$(date '+%H:%M:%S')] Starting MS-DETR sweep — full augmentation (15 trials x 30 epochs)"
export WANDB_AGENT_MAX_INITIAL_FAILURES=100
CUDA_VISIBLE_DEVICES=1 python yamin/experiment-msdetr/hparam_sweep.py --aug_mode full
echo "[$(date '+%H:%M:%S')] Sweep (full) complete. Best config:"
cat yamin/experiment-msdetr/best_sweep_config_msdetr_full.json
