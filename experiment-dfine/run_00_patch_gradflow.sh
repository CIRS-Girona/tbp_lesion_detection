#!/usr/bin/env bash
# ============================================================
# Step 0 — Patch gradient-flow logging into D-FINE (run ONCE)
# This patches det_engine.py to save per-step gradient CSVs/PNGs.
# Creates a backup at det_engine.py.itobos_bak — safe to restore.
# ============================================================
set -euo pipefail
source ~/code/iToBoS/aritra/dfine_code/dfine_env/bin/activate

ARITRA_SCRIPTS=~/code/iToBoS/aritra/dfine_code/itobos_dfine_code/scripts
DFINE_REPO=~/code/iToBoS/aritra/dfine_code/D-FINE

echo "=== Patching gradient-flow logging into D-FINE ==="
python "$ARITRA_SCRIPTS/patch_gradient_flow.py" \
  --dfine-repo "$DFINE_REPO"

echo ""
echo "Done. Gradient-flow PNGs/CSVs will be saved to:"
echo "  <output_dir>/gradient_flow/"
echo ""
echo "To restore the original file later:"
echo "  python $ARITRA_SCRIPTS/patch_gradient_flow.py --dfine-repo $DFINE_REPO --restore"
