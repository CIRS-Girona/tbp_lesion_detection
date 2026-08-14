# experiment-v12 — YOLOv12s Experiments

Self-contained experiment folder for YOLOv12s skin lesion detection.
Mirrors the structure of `yamin/` so every result is separately tracked.

---

## Folder Structure

```
yamin/experiment-v12/
├── hparam_sweep.py              # Bayesian HP sweep (handles noaug + full)
├── train_best.py                # 100-epoch full training from best HP config
├── evaluate.py                  # Threshold sweep evaluation (conf × IoU NMS)
├── sweep_config_v12_noaug.yaml  # Sweep config: no augmentation
├── sweep_config_v12_full.yaml   # Sweep config: full augmentation
├── run_sweep_noaug.sh           # Run noaug sweep (Step 1a)
├── run_sweep_full.sh            # Run full-aug sweep (Step 1b)
├── run_train_noaug.sh           # Train noaug best model (Step 2a)
├── run_train_full.sh            # Train full-aug best model (Step 2b)
├── run_evaluate_noaug.sh        # Evaluate noaug model (Step 3a)
├── run_evaluate_full.sh         # Evaluate full-aug model (Step 3b)
└── runs/
    ├── sweeps/
    │   ├── yolo12s_noaug/       # Sweep trial checkpoints (noaug)
    │   └── yolo12s_full/        # Sweep trial checkpoints (full)
    ├── best_model/
    │   ├── yolo12s_best_noaug_100ep/  # Best noaug model weights
    │   └── yolo12s_best_full_100ep/   # Best full-aug model weights
    └── evaluation/
        ├── threshold_sweep_val.csv    # Threshold sweep results
        └── threshold_sweep_test.csv
```

---

## Key Design Decisions

| Item | Decision | Reason |
|------|----------|--------|
| WandB project | `skin-lesion-detection` (same as v26) | Easy cross-model comparison |
| Sweep metric | F1 = 2·P·R / (P+R) | More clinically meaningful than mAP50 |
| Sweep trials | 25 per mode | Same as v26 sweeps |
| noaug mode | Only `fliplr` kept | Mirrors v26 ablation exactly |
| full mode | All aug params tunable | Let Bayesian search find optimal values |
| IoU sweep | Yes — NMS-based! | Unlike v26 (NMS-free), IoU WILL matter here |

---

## Path Logic

```
SCRIPT_DIR  = ~/code/iToBoS/yamin/experiment-v12/
YAMIN_DIR   = ~/code/iToBoS/yamin/           (SCRIPT_DIR.parent)
ITOBOS_ROOT = ~/code/iToBoS/                 (SCRIPT_DIR.parent.parent)
DATA_YAML   = ~/code/iToBoS/params.yaml
MODEL       = ~/code/iToBoS/yolo12s.pt
```

---

## Execution Order (tmux)

### Step 0 — Check yolo12s.pt is present

```bash
ls ~/code/iToBoS/yolo12s.pt
# If missing: python -c "from ultralytics import YOLO; YOLO('yolo12s.pt')"
```

### Step 1a — No-Aug Sweep (~18h, 25 trials)

```bash
tmux new-session -s sweep_v12_noaug
bash yamin/experiment-v12/run_sweep_noaug.sh 25
```

### Step 1b — Full-Aug Sweep (~18h, 25 trials) [run in parallel on different GPU if available]

```bash
tmux new-session -s sweep_v12_full
bash yamin/experiment-v12/run_sweep_full.sh 25
```

### Step 2a — Train best noaug model (after step 1a)

```bash
tmux new-session -s train_v12_noaug
bash yamin/experiment-v12/run_train_noaug.sh
```

### Step 2b — Train best full-aug model (after step 1b)

```bash
tmux new-session -s train_v12_full
bash yamin/experiment-v12/run_train_full.sh
```

### Step 3a — Evaluate noaug model

```bash
tmux new-session -s eval_v12_noaug
bash yamin/experiment-v12/run_evaluate_noaug.sh
```

### Step 3b — Evaluate full-aug model

```bash
tmux new-session -s eval_v12_full
bash yamin/experiment-v12/run_evaluate_full.sh
```

---

## Output Files

After all steps:

| File | Contents |
|------|----------|
| `best_sweep_config_yolo12s_noaug.json` | Best HP config from noaug sweep |
| `best_sweep_config_yolo12s_full.json` | Best HP config from full sweep |
| `runs/best_model/yolo12s_best_noaug_100ep/weights/best.pt` | Trained noaug model |
| `runs/best_model/yolo12s_best_full_100ep/weights/best.pt` | Trained full-aug model |
| `runs/evaluation/threshold_sweep_val.csv` | Val threshold sweep table |
| `runs/evaluation/threshold_sweep_test.csv` | Test threshold sweep table |

---

## Comparison with YOLOv26s

| Experiment | v26s noaug | v26s full | v12s noaug | v12s full |
|------------|-----------|-----------|-----------|-----------|
| WandB run  | ✅ Done   | ✅ Done   | ⏳ Pending | ⏳ Pending |
| Best F1    | —         | —         | TBD        | TBD        |
| NMS-free   | ✅ Yes    | ✅ Yes    | ❌ No      | ❌ No      |
| IoU effect | None      | None      | **Yes**    | **Yes**    |
