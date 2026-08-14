# D-FINE Experiment Commands — Yamin
## Folder: `~/code/iToBoS/yamin/experiment-dfine`

---

## 0. Upload files to server (run locally on your Windows machine)

```powershell
# From PowerShell:
scp -P 2222 -r "C:\Users\Muhammad\Downloads\projects-temp\perception\skin-lesion-detection\further-experiments\dfine-experiments\." hope2@falcon.infoblitz.net:~/code/iToBoS/yamin/experiment-dfine/

# Make scripts executable:
ssh -p 2222 hope2@falcon.infoblitz.net "chmod +x ~/code/iToBoS/yamin/experiment-dfine/run_*.sh"
```

---

## 1. Activate environment & verify setup

```bash
source ~/code/iToBoS/aritra/dfine_code/dfine_env/bin/activate
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"
pip install wandb --quiet
wandb login
```

---

## 2. Step 0 — Patch gradient-flow logging (ONCE ONLY)

```bash
cd ~/code/iToBoS/yamin/experiment-dfine
bash run_00_patch_gradflow.sh
```

> Patches `det_engine.py` to log gradient magnitudes every 200 steps.
> Creates a backup. To restore: add `--restore` flag.

---

## 3. Step 1 — Bayesian sweep (20 trials × 30 epochs each)

```bash
# Create the sweep:
source ~/code/iToBoS/aritra/dfine_code/dfine_env/bin/activate
cd ~/code/iToBoS/yamin/experiment-dfine
wandb sweep sweep_config.yaml
# Copy the printed ID e.g.: yamin/skin-lesion-dfine-sweep/abc123def
```

Edit `run_01_sweep.sh` — set `SWEEP_ID="yamin/.../abc123def"` — then:

```bash
tmux new -s dfine_sweep
source ~/code/iToBoS/aritra/dfine_code/dfine_env/bin/activate
cd ~/code/iToBoS/yamin/experiment-dfine
bash run_01_sweep.sh
# Ctrl+B D to detach — leave overnight (~12h total)
```

**After sweep:** get best lr, backbone_lr_ratio, weight_decay from wandb.ai
Update `BEST_LR`, `BEST_BB_LR`, `BEST_WD` in `run_02_train_fullaug.sh` and `run_03_train_noaug.sh`.

> **Skip sweep option:** defaults already set to Aritra's HPs (lr=5e-5, bb_lr=2.5e-5, wd=1e-4).

---

## 4. Step 2 — Full 100-epoch training

```bash
# Full aug (tmux session 1):
tmux new -s dfine_fullaug
source ~/code/iToBoS/aritra/dfine_code/dfine_env/bin/activate
cd ~/code/iToBoS/yamin/experiment-dfine
bash run_02_train_fullaug.sh
# Ctrl+B D

# No aug (tmux session 2, or after fullaug finishes):
tmux new -s dfine_noaug
source ~/code/iToBoS/aritra/dfine_code/dfine_env/bin/activate
cd ~/code/iToBoS/yamin/experiment-dfine
bash run_03_train_noaug.sh
```

> Expected: ~6–7h each on A100. Checkpoints every 5 epochs.
> Best checkpoint: `results/training/full/.../best_stg2.pth`

---

## 5. Step 3 — Loss ablation (Hayat's analysis — ~10.5h total)

```bash
tmux new -s dfine_ablation
source ~/code/iToBoS/aritra/dfine_code/dfine_env/bin/activate
cd ~/code/iToBoS/yamin/experiment-dfine
bash run_04_ablation_losses.sh
```

Three ablations × 50 epochs:

| Run | What changes | Scientific question |
|-----|-------------|---------------------|
| `ablation_fgl0` | FGL weight = 0 | Does removing FGL degrade mAP50? (proves D-FINE innovation) |
| `ablation_fgl_high` | FGL weight = 0.30 (2×) | Is stronger distribution supervision better? |
| `ablation_ddf0` | DDF weight = 0 | Does inter-layer distillation matter here? |

---

## 6. Step 4 — Threshold sweep + evaluation

```bash
source ~/code/iToBoS/aritra/dfine_code/dfine_env/bin/activate
cd ~/code/iToBoS/yamin/experiment-dfine
bash run_05_evaluate_all.sh
```

Outputs per model (in `results/evaluation/<label>/test/`):
- `threshold_sweep_test.csv` — 11×9 F1/P/R grid
- `pr_curve_test.png` — PR curve
- `best_threshold_summary.json` — best F1, conf, NMS-IoU

---

## 7. Step 5 — Loss curves + gradient flow plots

```bash
source ~/code/iToBoS/aritra/dfine_code/dfine_env/bin/activate
cd ~/code/iToBoS/yamin/experiment-dfine
bash run_06_plot_losses.sh
```

Outputs per model (in `results/training/<aug>/<run>/plots/`):
- `training_loss_curves_all.png` — all losses together
- `training_loss_fgl.png`, `training_loss_ddf.png`, etc.
- Gradient flow PNGs in `gradient_flow/` subdirectory

---

## 8. Monitoring commands

```bash
# GPU usage
watch -n 2 nvidia-smi

# Tail running training log
tail -f ~/code/iToBoS/yamin/experiment-dfine/results/training/full/dfine_s_fullaug_sweep_100ep/train_stdout.log

# List tmux sessions
tmux ls

# Reattach
tmux attach -t dfine_fullaug
```

---

## 9. Expected results for the paper

| Experiment | Key metric | Paper use |
|-----------|-----------|-----------|
| fullaug_sweep 100ep | mAP50 > 0.674, F1, conf threshold | Update Tab. 1 main results |
| noaug_sweep 100ep | mAP50, F1 | Update Tab. 1 no-aug row |
| ablation_fgl0 | mAP50 drop | Tab. N: loss ablation table |
| ablation_fgl_high | mAP50 change | Tab. N: loss ablation |
| ablation_ddf0 | mAP50 drop | Tab. N: loss ablation |
| gradient flow PNGs | Figure | Sec: Gradient Analysis |
| loss curves | Figure | Sec: Loss Function Analysis |

---

## Notes

- Dataset: Aritra's COCO split at `~/code/iToBoS/aritra/dfine_results/dataset_coco`
- D-FINE source: Aritra's at `~/code/iToBoS/aritra/dfine_code/D-FINE`
- All output: `~/code/iToBoS/yamin/experiment-dfine/results/`
- CUDA OOM? Reduce `--batch 4` and halve lr accordingly
