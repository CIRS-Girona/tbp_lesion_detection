#!/usr/bin/env bash
# run_sweep_noaug.sh — MS-DETR sweep, no augmentation
# Run from ~/code/iToBoS/:
#   bash yamin/experiment-msdetr/run_sweep_noaug.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$REPO_ROOT/yamin/msdetr-venv/bin/activate"
[[ -f "$VENV" ]] || { echo "[ERROR] msdetr-venv not found. Run run_setup.sh first."; exit 1; }
source "$VENV"
cd "$REPO_ROOT"
echo "[$(date '+%H:%M:%S')] Starting MS-DETR sweep — no augmentation (15 trials x 30 epochs)"
export WANDB_AGENT_MAX_INITIAL_FAILURES=100
CUDA_VISIBLE_DEVICES=0 python yamin/experiment-msdetr/hparam_sweep.py --aug_mode noaug
echo "[$(date '+%H:%M:%S')] Sweep (noaug) complete. Best config:"
cat yamin/experiment-msdetr/best_sweep_config_msdetr_noaug.json
