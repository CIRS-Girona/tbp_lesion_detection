# experiment-v8: YOLOv8s Skin Lesion Detection

YOLOv8s baseline experiment on the iToBoS dataset.
Same full pipeline as experiment-v11/v12/v26: Bayesian sweep → full training → threshold sweep evaluation.

**Assigned to:** Muhammad Yamin  
**Goal:** Establish whether the older YOLOv8s baseline outperforms or underperforms the v11/v12/v26 models when trained with identical methodology.

---

## Files

| File | Purpose |
|---|---|
| `hparam_sweep.py` | WandB Bayesian HP sweep agent (25 trials × 40 epochs) |
| `train_best.py` | Full 100-epoch training using best sweep config |
| `evaluate.py` | 11×9 conf × NMS IoU threshold sweep on val + test |
| `sweep_config_v8_full.yaml` | WandB sweep config (full augmentation) |
| `sweep_config_v8_noaug.yaml` | WandB sweep config (no augmentation) |
| `run_sweep_full.sh` | Run full-aug sweep |
| `run_sweep_noaug.sh` | Run noaug sweep |
| `run_train_full.sh` | Run full-aug 100-epoch training |
| `run_train_noaug.sh` | Run noaug 100-epoch training |
| `run_evaluate_full.sh` | Run evaluation for full-aug model |
| `run_evaluate_noaug.sh` | Run evaluation for noaug model |

---

## Execution Order

Run all commands from `~/code/iToBoS/` (the iToBoS root directory).

### Step 1: Bayesian Sweep (both conditions in parallel on different GPUs)

```bash
# On GPU 0 (full aug):
bash yamin/experiment-v8/run_sweep_full.sh

# On GPU 1 (no aug) -- run simultaneously in a separate terminal:
bash yamin/experiment-v8/run_sweep_noaug.sh
```

Each sweep runs 25 trials of 40 epochs. Best config saved automatically to:
- `best_sweep_config_yolov8s_full.json`
- `best_sweep_config_yolov8s_noaug.json`

### Step 2: Full Training (after sweeps complete)

```bash
bash yamin/experiment-v8/run_train_full.sh
bash yamin/experiment-v8/run_train_noaug.sh
```

100 epochs, patience=20, AdamW, cosine LR. Weights saved to:
- `runs/best_model/yolov8s_best_full_100ep/weights/best.pt`
- `runs/best_model/yolov8s_best_noaug_100ep/weights/best.pt`

### Step 3: Threshold Sweep Evaluation (after training)

```bash
bash yamin/experiment-v8/run_evaluate_full.sh
bash yamin/experiment-v8/run_evaluate_noaug.sh
```

Sweeps 11 confidence thresholds × 9 NMS IoU thresholds on both val and test sets.
Results saved to WandB and local CSV.

---

## Key Differences from v11/v12/v26

- Model weight: `yolov8s.pt` (downloads automatically on first run)
- YOLOv8s is an older architecture (C2 blocks, anchor-free NMS-based)
- Expected: slightly lower performance than v11/v12/v26 — confirms architectural progress
- Parameter count: ~11.2M (slightly larger than v11s at 9.43M)

---

## Results Destination

Record final test results in the master comparison table:
- F1, mAP50, Precision, Recall at conf=0.20
- Optimal NMS IoU from threshold sweep
- Training time

Compare against v11 (F1=0.631), v12 (F1=0.640), v26 (F1=0.634).
