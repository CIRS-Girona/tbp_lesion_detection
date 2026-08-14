# Server Setup Guide — experiment-v12
# Run these commands from your LOCAL PowerShell

# ── Step 1: Upload experiment-v12 folder to server ────────────────────────────
# Creates the folder and uploads all scripts in one go

scp -P 2222 -r "C:\Users\Muhammad\Downloads\projects-temp\perception\skin-lesion-detection\yamin\experiment-v12" hope2@falcon.infoblitz.net:~/code/iToBoS/yamin/


# ── Step 2: SSH into server ───────────────────────────────────────────────────
ssh -p 2222 hope2@falcon.infoblitz.net


# ── On the server: check prerequisites ───────────────────────────────────────
cd ~/code/iToBoS

# Check yolo12s.pt exists (auto-downloads if missing):
python -c "from ultralytics import YOLO; YOLO('yolo12s.pt')"

# Verify paths are correct:
python -c "
from pathlib import Path
s = Path('yamin/experiment-v12/hparam_sweep.py').resolve().parent
print('SCRIPT_DIR  :', s)
print('ITOBOS_ROOT :', s.parent.parent)
print('DATA_YAML   :', s.parent.parent / 'params.yaml')
print('MODEL       :', s.parent.parent / 'yolo12s.pt')
"


# ── Step 3a: Start NO-AUG SWEEP in tmux ──────────────────────────────────────
tmux new-session -d -s sweep_v12_noaug
tmux send-keys -t sweep_v12_noaug "cd ~/code/iToBoS && bash yamin/experiment-v12/run_sweep_noaug.sh 25" Enter

# Attach to watch progress:
tmux attach -t sweep_v12_noaug
# Detach: Ctrl+B, then D


# ── Step 3b: Start FULL-AUG SWEEP in separate tmux ──────────────────────────
# (Run AFTER the noaug sweep finishes, or simultaneously if GPU allows)
tmux new-session -d -s sweep_v12_full
tmux send-keys -t sweep_v12_full "cd ~/code/iToBoS && bash yamin/experiment-v12/run_sweep_full.sh 25" Enter

tmux attach -t sweep_v12_full


# ── Step 4a: Train best noaug model (after sweep 3a) ────────────────────────
tmux new-session -d -s train_v12_noaug
tmux send-keys -t train_v12_noaug "cd ~/code/iToBoS && bash yamin/experiment-v12/run_train_noaug.sh" Enter

tmux attach -t train_v12_noaug


# ── Step 4b: Train best full-aug model (after sweep 3b) ─────────────────────
tmux new-session -d -s train_v12_full
tmux send-keys -t train_v12_full "cd ~/code/iToBoS && bash yamin/experiment-v12/run_train_full.sh" Enter

tmux attach -t train_v12_full


# ── Step 5: Evaluate (after training) ────────────────────────────────────────
tmux new-session -d -s eval_v12_noaug
tmux send-keys -t eval_v12_noaug "cd ~/code/iToBoS && bash yamin/experiment-v12/run_evaluate_noaug.sh" Enter

tmux new-session -d -s eval_v12_full
tmux send-keys -t eval_v12_full "cd ~/code/iToBoS && bash yamin/experiment-v12/run_evaluate_full.sh" Enter


# ── Useful tmux commands ──────────────────────────────────────────────────────
# List all sessions:      tmux ls
# Attach to session:      tmux attach -t <name>
# Detach from session:    Ctrl+B, then D
# Kill a session:         tmux kill-session -t <name>


# ── Download results when done ────────────────────────────────────────────────
# Run from LOCAL PowerShell:
scp -P 2222 -r hope2@falcon.infoblitz.net:~/code/iToBoS/yamin/experiment-v12/runs "C:\Users\Muhammad\Downloads\projects-temp\perception\skin-lesion-detection\yamin\experiment-v12\"
scp -P 2222 "hope2@falcon.infoblitz.net:~/code/iToBoS/yamin/experiment-v12/best_sweep_config_yolo12s_noaug.json" "C:\Users\Muhammad\Downloads\projects-temp\perception\skin-lesion-detection\yamin\experiment-v12\"
scp -P 2222 "hope2@falcon.infoblitz.net:~/code/iToBoS/yamin/experiment-v12/best_sweep_config_yolo12s_full.json" "C:\Users\Muhammad\Downloads\projects-temp\perception\skin-lesion-detection\yamin\experiment-v12\"
