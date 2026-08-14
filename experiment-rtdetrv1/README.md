# experiment-rtdetr — RT-DETR-L on iToBoS

RT-DETR (Real-Time Detection Transformer) benchmark for the iToBoS skin lesion
detection paper. Mirrors the structure of Yamin's `experiment-v11/v12/v26`.

**Owner:** Praveen Kumar Murali
**Model:** RT-DETR-L (Ultralytics), ~32M params, ResNet-50 backbone, NMS-free
**Framework:** Ultralytics 8.4.37

## Trial result (13 June 2026 — FEASIBLE)

20-epoch quick run, default HPs, full aug, validation set:

| Metric | RT-DETR @ 20ep | YOLOv12s @ 100ep |
|---|---|---|
| F1 | **0.6438** | 0.640 |
| mAP50 | **0.6719** | 0.652 |
| Precision | 0.6818 | 0.680 |
| Recall | 0.6099 | 0.604 |

Training time: 1.76 hrs on one A100. RT-DETR already matches the best YOLO model
at 20 untuned epochs.

## Files

| File | Purpose |
|---|---|
| `quick_train.py` | One-off 20-epoch feasibility run (already done) |
| `hparam_sweep.py` | Bayesian sweep (40 trials × 40 ep), saves best config JSON |
| `sweep_config_rtdetr_full.yaml` | Sweep search space — full augmentation |
| `sweep_config_rtdetr_noaug.yaml` | Sweep search space — no augmentation |
| `train_best.py` | Full 100-epoch training using best sweep config |
| `evaluate.py` | Confidence threshold sweep on val + test, logs to WandB |
| `run_*.sh` | Shell wrappers for each step |
| `SERVER_COMMANDS.sh` | Full workflow: upload, run, download |

## Key differences from the YOLO experiments

1. **No box/cls/dfl tuning.** RT-DETR's loss in Ultralytics uses fixed internal
   gains (class=1, bbox=5, giou=2). Those args do not affect RT-DETR, so the
   sweep does not tune them.
2. **No NMS / IoU tuning.** RT-DETR is end-to-end (Hungarian matching). The IoU
   threshold has no effect at inference, like YOLOv26s. `evaluate.py` fixes
   iou=0.50 by default; use `--iou_sweep` to demonstrate flatness.
3. **Lower learning rate.** Transformer detectors need smaller LR than YOLO:
   sweep range is 1e-5 to 1e-3 (vs YOLO's up to 1e-2).

## Workflow (run on server from ~/code/iToBoS/)

```bash
# 1. Sweep (find best HPs) — full aug, 40 trials
tmux new-session -d -s rtdetr_sweep_full
tmux send-keys -t rtdetr_sweep_full \
  "cd ~/code/iToBoS && bash praveen/experiment-rtdetr/run_sweep_full.sh 40" Enter

# 2. Train best config for 100 epochs
bash praveen/experiment-rtdetr/run_train_full.sh 0

# 3. Evaluate (threshold sweep on test set)
bash praveen/experiment-rtdetr/run_evaluate_full.sh 0

# Repeat 1-3 with the _noaug variants for the no-augmentation condition.
```

## Metric conventions (paper rules)

- F1 / Precision / Recall reported at the optimal confidence threshold.
- mAP50 computed at conf=0.001 (full PR curve) — handled in `evaluate.py`.
- Inference speed (FPS) is NOT measured here. Muhammad runs all speed benchmarks
  on his laptop CPU for fairness. Share `best.pt` with him.
