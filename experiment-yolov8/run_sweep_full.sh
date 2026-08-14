#!/bin/bash
# ============================================================
# run_sweep_full.sh
# Launch YOLOv8s HP sweep -- FULL AUGMENTATION
# Location: yamin/experiment-v8/
#
# Run from iToBoS root:
#   bash yamin/experiment-v8/run_sweep_full.sh           # 25 trials
#   bash yamin/experiment-v8/run_sweep_full.sh 15        # custom count
# ============================================================

set -e

WANDB_API_KEY="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"
WANDB_PROJECT="skin-lesion-detection"
SWEEP_COUNT="${1:-25}"

cd ~/code/iToBoS
source ~/code/iToBoS/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY="${WANDB_API_KEY}"

echo "=============================================="
echo "  iToBoS | YOLOv8s Sweep -- FULL AUGMENTATION"
echo "=============================================="
echo "  Project : ${WANDB_PROJECT}"
echo "  Trials  : ${SWEEP_COUNT}"
echo "  GPU     : CUDA_VISIBLE_DEVICES=0"
echo "  Config  : yamin/experiment-v8/sweep_config_v8_full.yaml"
echo "=============================================="
echo ""

echo "[1/2] Creating WandB sweep..."
SWEEP_OUTPUT=$(wandb sweep \
    --project "${WANDB_PROJECT}" \
    yamin/experiment-v8/sweep_config_v8_full.yaml 2>&1)
echo "${SWEEP_OUTPUT}"

AGENT_CMD=$(echo "${SWEEP_OUTPUT}" | grep -oE "wandb agent [^ ]+")
if [ -z "$AGENT_CMD" ]; then
    echo "ERROR: Could not parse sweep agent command."
    echo "       Copy the 'wandb agent ...' line above and run manually."
    exit 1
fi

SWEEP_PATH=$(echo "${AGENT_CMD}" | awk '{print $NF}')
echo ""
echo "  Sweep: ${SWEEP_PATH}"

echo "${SWEEP_PATH}"  > yamin/experiment-v8/last_sweep_id_full.txt
echo "${AGENT_CMD}"   > yamin/experiment-v8/last_agent_cmd_full.txt
echo "  (Saved to yamin/experiment-v8/last_sweep_id_full.txt)"

echo ""
echo "[2/2] Starting wandb agent (${SWEEP_COUNT} trials)..."
echo "      Ctrl+C stops the agent -- completed trials are already saved."
echo ""

wandb agent --count "${SWEEP_COUNT}" "${SWEEP_PATH}"

echo ""
echo "=============================================="
echo "  Sweep complete!"
echo "  Best config: yamin/experiment-v8/best_sweep_config_yolov8s_full.json"
echo "  Next step  : bash yamin/experiment-v8/run_train_full.sh"
echo "=============================================="
