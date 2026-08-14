# YOLOv11s Experiment — iToBoS Skin Lesion Detection
**Week 5 | May 2026 | Author: Yamin**

Third architecture in the comparative study: YOLOv11s (standard NMS-based YOLO baseline).
Identical experimental pipeline to experiment-v12 and v26.

## Architecture Context
- **YOLOv26s** — NMS-free (Week 3)
- **YOLOv12s** — NMS-based, attention backbone (Week 4)
- **YOLOv11s** — NMS-based, standard YOLO baseline (Week 5 ← THIS)

YOLOv11s serves as the "older, standard" YOLO baseline. Comparing all three gives a clear
picture of the progression from standard → attention → NMS-free architectures.

## Pipeline (same as v12)

```
Step 1: HP Sweeps (25 trials × 40 epochs each)
  → run_sweep_full.sh   (GPU 3, full augmentation)
  → run_sweep_noaug.sh  (GPU 1, no augmentation)

Step 2: Full Training (100 epochs)
  → run_train_full.sh
  → run_train_noaug.sh

Step 3: Threshold Evaluation
  → run_evaluate_full.sh
  → run_evaluate_noaug.sh
```

## File Structure

```
experiment-v11/
├── hparam_sweep.py              # WandB Bayesian sweep (--model yolo11s.pt --aug_mode [full|noaug])
├── train_best.py                # Full 100-epoch training with best sweep HPs
├── evaluate.py                  # Threshold sweep (11 conf × 9 NMS IoU = 99 combos)
├── sweep_config_v11_full.yaml   # Sweep config: full augmentation
├── sweep_config_v11_noaug.yaml  # Sweep config: no augmentation
├── run_sweep_full.sh            # Step 1a: Launch full-aug sweep
├── run_sweep_noaug.sh           # Step 1b: Launch noaug sweep
├── run_train_full.sh            # Step 2a: Train full-aug model
├── run_train_noaug.sh           # Step 2b: Train noaug model
├── run_evaluate_full.sh         # Step 3a: Evaluate full-aug model
├── run_evaluate_noaug.sh        # Step 3b: Evaluate noaug model
├── best_sweep_config_yolo11s_full.json    # Auto-saved after sweep
├── best_sweep_config_yolo11s_noaug.json   # Auto-saved after sweep
└── runs/
    ├── sweeps/
    │   ├── yolo11s_full/
    │   └── yolo11s_noaug/
    ├── best_model/
    │   ├── yolo11s_best_full_100ep/weights/best.pt
    │   └── yolo11s_best_noaug_100ep/weights/best.pt
    └── evaluation/
        ├── full/
        │   ├── threshold_sweep_val.csv
        │   └── threshold_sweep_test.csv
        └── noaug/
            ├── threshold_sweep_val.csv
            └── threshold_sweep_test.csv
```

## Run Order

```bash
# From ~/code/iToBoS

# Step 1 — run both sweeps in parallel (different GPUs)
tmux new -s sweep_full
bash yamin/experiment-v11/run_sweep_full.sh    # GPU 3

tmux new -s sweep_noaug
bash yamin/experiment-v11/run_sweep_noaug.sh   # GPU 1

# Step 2 — run trainings after sweeps complete
bash yamin/experiment-v11/run_train_full.sh
bash yamin/experiment-v11/run_train_noaug.sh

# Step 3 — evaluate both models
bash yamin/experiment-v11/run_evaluate_full.sh
bash yamin/experiment-v11/run_evaluate_noaug.sh
```

## Expected Results
Based on v12 patterns, expect:
- Full aug F1 > No aug F1 (consistent with v12: +1.4 pp, v26: +1.9 pp)
- No aug model trains faster (early stopping from diversity exhaustion)
- conf=0.20 optimal (confirmed for v12 and v26)
- NMS IoU spread should be similarly negligible (non-overlapping lesions)

## Key Comparison Table (fill in after experiments)

| Model | Aug | F1 (test) | mAP50 (test) | Precision | Recall |
|-------|-----|-----------|--------------|-----------|--------|
| YOLOv12s | Full | 0.640 | 0.652 | 0.680 | 0.604 |
| YOLOv26s | Full | 0.636 | 0.639 | 0.667 | 0.608 |
| **YOLOv11s** | **Full** | **TBD** | **TBD** | **TBD** | **TBD** |
| YOLOv12s | None | 0.630 | 0.624 | 0.661 | 0.620 |
| YOLOv26s | None | 0.617 | 0.606 | 0.655 | 0.583 |
| **YOLOv11s** | **None** | **TBD** | **TBD** | **TBD** | **TBD** |

## WandB
- Project: `skin-lesion-detection`
- Entity: `myamin-cs-universitat-de-girona`
- All runs tagged `experiment-v11`
