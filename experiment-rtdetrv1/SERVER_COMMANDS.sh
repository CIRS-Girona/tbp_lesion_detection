# ==============================================================
# SERVER_COMMANDS.sh
# RT-DETR experiment — full workflow guide
# Upload/download commands run from your LOCAL terminal.
# Sweep/train/eval commands run ON THE SERVER.
# ==============================================================


# ── Step 0: Re-upload updated folder (after local edits) ──────────────────────
# (Safe — adds/updates files, does not touch running jobs or existing runs/)

scp -P 2222 -r ~/Documents/rtdetr/experiment-rtdetr \
    hope2@falcon.infoblitz.net:~/code/iToBoS/praveen/


# ── Step 1: SSH in and verify ─────────────────────────────────────────────────

ssh -p 2222 hope2@falcon.infoblitz.net

cd ~/code/iToBoS
source .venv/bin/activate
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.free \
           --format=csv,noheader


# ============================================================
#  FULL PIPELINE — run each step on the server
#  (repeat with _noaug variants for the no-aug condition)
# ============================================================

# ── Step 2: Bayesian sweep — FULL AUG (40 trials, ~ many hours) ───────────────
tmux new-session -d -s rtdetr_sweep_full
tmux send-keys -t rtdetr_sweep_full \
    "cd ~/code/iToBoS && bash praveen/experiment-rtdetr/run_sweep_full.sh 40" Enter
tmux attach -t rtdetr_sweep_full      # detach: Ctrl+B then D

# ── Step 2b: Bayesian sweep — NO AUG (run on a different free GPU) ────────────
# First edit the --device in sweep_config_rtdetr_noaug.yaml if needed, OR run
# after the full sweep finishes on GPU 0.
tmux new-session -d -s rtdetr_sweep_noaug
tmux send-keys -t rtdetr_sweep_noaug \
    "cd ~/code/iToBoS && bash praveen/experiment-rtdetr/run_sweep_noaug.sh 40" Enter

# ── Step 3: Full 100-epoch training (after sweeps) ───────────────────────────
# Pass GPU index as the argument.
tmux new-session -d -s rtdetr_train_full
tmux send-keys -t rtdetr_train_full \
    "cd ~/code/iToBoS && bash praveen/experiment-rtdetr/run_train_full.sh 0" Enter

tmux new-session -d -s rtdetr_train_noaug
tmux send-keys -t rtdetr_train_noaug \
    "cd ~/code/iToBoS && bash praveen/experiment-rtdetr/run_train_noaug.sh 1" Enter

# ── Step 4: Evaluation (threshold sweep on val + test) ───────────────────────
tmux new-session -d -s rtdetr_eval_full
tmux send-keys -t rtdetr_eval_full \
    "cd ~/code/iToBoS && bash praveen/experiment-rtdetr/run_evaluate_full.sh 0" Enter

tmux new-session -d -s rtdetr_eval_noaug
tmux send-keys -t rtdetr_eval_noaug \
    "cd ~/code/iToBoS && bash praveen/experiment-rtdetr/run_evaluate_noaug.sh 1" Enter


# ── tmux cheatsheet ───────────────────────────────────────────────────────────
# List:    tmux ls
# Attach:  tmux attach -t <name>
# Detach:  Ctrl+B then D
# Kill:    tmux kill-session -t <name>


# ============================================================
#  DOWNLOAD RESULTS (run from LOCAL terminal)
# ============================================================

# Best sweep config JSONs (small)
scp -P 2222 \
    "hope2@falcon.infoblitz.net:~/code/iToBoS/praveen/experiment-rtdetr/best_sweep_config_rtdetr-l_*.json" \
    ~/Documents/rtdetr/experiment-rtdetr/

# Best model runs (weights + curves + results.csv)
scp -P 2222 -r \
    "hope2@falcon.infoblitz.net:~/code/iToBoS/praveen/experiment-rtdetr/runs/best_model" \
    ~/Documents/rtdetr/experiment-rtdetr/runs/

# Evaluation results (threshold sweep CSVs + plots)
scp -P 2222 -r \
    "hope2@falcon.infoblitz.net:~/code/iToBoS/praveen/experiment-rtdetr/runs/evaluation" \
    ~/Documents/rtdetr/experiment-rtdetr/runs/


# ── Share weights with Muhammad for CPU inference benchmark ───────────────────
# He needs the best.pt files (full + noaug) to run speed tests on his laptop.
