# experiment-v26 — YOLOv26s Experiments

Self-contained experiment folder for YOLOv26s skin lesion detection.
Mirrors the structure of `yamin/` so every result is separately tracked.

---

## Folder Structure

```
yamin/experiment-v26/
├── hparam_sweep.py              # Bayesian HP sweep (handles noaug + full)
├── train_best.py                # 100-epoch full training from best HP config
├── evaluate.py                  # Threshold sweep evaluation (conf × IoU NMS)
├── sweep_config_v26s_noaug.yaml # Sweep config: no augmentation
├── sweep_config_v26s_full.yaml  # Sweep config: full augmentation
├── run_sweep_noaug.sh           # Run noaug sweep (Step 1a)
├── run_sweep_full.sh            # Run full-aug sweep (Step 1b)
├── run_train_noaug.sh           # Train noaug best model (Step 2a)
├── run_train_full.sh            # Train full-aug best model (Step 2b)
├── run_evaluate_noaug.sh        # Evaluate noaug model (Step 3a)
├── run_evaluate_full.sh         # Evaluate full-aug model (Step 3b)
└── runs/
    ├── sweeps/
    │   ├── yolo26s_noaug/       # Sweep trial checkpoints (noaug)
    │   └── yolo26s_full/        # Sweep trial checkpoints (full)
    ├── best_model/
    │   ├── yolo26s_best_noaug_100ep/  # Best noaug model weights
    │   └── yolo26s_best_full_100ep/   # Best full-aug model weights
    └── evaluation/
        ├── threshold_sweep_val.csv    # Threshold sweep results
        └── threshold_sweep_test.csv
```

---

## Key Design Decisions

| Item | Decision | Reason |
|------|----------|--------|
| WandB project | `skin-lesion-detection` (same as v12) | Easy cross-model comparison |
| Sweep metric | F1 = 2·P·R / (P+R) | More clinically meaningful than mAP50 |
| Sweep trials | 25 per mode | Standardised search space budget |
| noaug mode | Only `fliplr` kept | Mirrors v12 ablation exactly |
| full mode | All aug params tunable | Let Bayesian search find optimal values |
| IoU sweep | Yes — NMS-free logic | Swiped to experimentally confirm NMS flatlines |

---

## Path Logic

```
SCRIPT_DIR  = ~/code/iToBoS/yamin/experiment-v26/
YAMIN_DIR   = ~/code/iToBoS/yamin/           (SCRIPT_DIR.parent)
ITOBOS_ROOT = ~/code/iToBoS/                 (SCRIPT_DIR.parent.parent)
DATA_YAML   = ~/code/iToBoS/params.yaml
MODEL       = ~/code/iToBoS/yolo26s.pt
```

---

## Execution Order (tmux)

### Step 0 — Check yolo26s.pt is present

```bash
ls ~/code/iToBoS/yolo26s.pt
# If missing: python -c "from ultralytics import YOLO; YOLO('yolo26s.pt')"
```

### Step 1a — No-Aug Sweep (~18h, 25 trials)

```bash
tmux new-session -s sweep_v26_noaug
bash yamin/experiment-v26/run_sweep_noaug.sh 25
```

### Step 1b — Full-Aug Sweep (~18h, 25 trials)

```bash
tmux new-session -s sweep_v26_full
bash yamin/experiment-v26/run_sweep_full.sh 25
```

### Step 2a — Train best noaug model (after step 1a)

```bash
tmux new-session -s train_v26_noaug
bash yamin/experiment-v26/run_train_noaug.sh
```

### Step 2b — Train best full-aug model (after step 1b)

```bash
tmux new-session -s train_v26_full
bash yamin/experiment-v26/run_train_full.sh
```

### Step 3a — Evaluate noaug model

```bash
tmux new-session -s eval_v26_noaug
bash yamin/experiment-v26/run_evaluate_noaug.sh
```

### Step 3b — Evaluate full-aug model

```bash
tmux new-session -s eval_v26_full
bash yamin/experiment-v26/run_evaluate_full.sh
```

---

## Output Files

After all steps:

| File | Contents |
|------|----------|
| `best_sweep_config_yolo26s_noaug.json` | Best HP config from noaug sweep |
| `best_sweep_config_yolo26s_full.json` | Best HP config from full sweep |
| `runs/best_model/yolo26s_best_noaug_100ep/weights/best.pt` | Trained noaug model |
| `runs/best_model/yolo26s_best_full_100ep/weights/best.pt` | Trained full-aug model |
| `runs/evaluation/threshold_sweep_val.csv` | Val threshold sweep table |
| `runs/evaluation/threshold_sweep_test.csv` | Test threshold sweep table |
