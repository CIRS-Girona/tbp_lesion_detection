# experiment-solomon — Solomon Baseline (YOLOv8s, Default HPs)

Reproduces Solomon's Kaggle approach on the iToBoS data using our evaluation
protocol (11×9 threshold sweep), so his results are directly comparable to our
YOLO v8/v11/v12/v26 models.

## What this experiment does

| Step | Script | Purpose |
|---|---|---|
| 1 | `train.py --aug_mode full` | YOLOv8s with Solomon's default YOLO augmentation |
| 2 | `evaluate.py --name full` | Default conf (0.325) + 11×9 sweep on val & test |
| 3 | `train.py --aug_mode noaug` | Ablation: flip-only augmentation |
| 4 | `evaluate.py --name noaug` | Same evaluation on no-aug model |

## Key differences from Solomon's original

| Setting | Solomon (Kaggle) | This experiment |
|---|---|---|
| Model | YOLOv8 (unspecified size) | **YOLOv8s** (matches our comparison) |
| Data split | Challenge data | iToBoS blurred split (same as other models) |
| HP tuning | None (default YOLO) | None (default YOLO — faithful reproduction) |
| Evaluation | Kaggle metric | F1/P/R/mAP50 via our 11×9 sweep |
| MedViT 2nd stage | Yes (in pipeline) | **Not used** — YOLO detector only |
| Data preprocessing | Background blur | ✅ Same blurred dataset already in use |

> **Why no MedViT?** The MedViT second stage in Solomon's `test_models.py` is a
> classifier that filters YOLO false positives. It requires a separately trained
> classification model (weights not available). The threshold sweep replaces this
> post-processing: by tuning conf, we achieve the same false-positive reduction
> effect in a reproducible and comparable way.

## Paper reporting

Two rows will be added to the results table:

| Model | Aug | Conf | F1 | P | R | mAP50 |
|---|---|---|---|---|---|---|
| Solomon YOLOv8s | Full | 0.325 (original) | TBD | TBD | TBD | TBD |
| Solomon YOLOv8s | Full | tuned (sweep) | TBD | TBD | TBD | TBD |

## How to run

```bash
# On server — from ~/code/iToBoS/
tmux new-session -d -s solomon -n train
tmux send-keys -t solomon:train \
  "cd ~/code/iToBoS && source .venv/bin/activate && bash yamin/experiment-solomon/run_solomon.sh" Enter
tmux attach -t solomon
```

## Output structure

```
yamin/experiment-solomon/
├── runs/
│   ├── best_model/
│   │   ├── solomon_yolov8s_full_100ep/weights/best.pt
│   │   └── solomon_yolov8s_noaug_100ep/weights/best.pt
│   └── evaluation/
│       ├── full/
│       │   ├── threshold_sweep_val.csv
│       │   └── threshold_sweep_test.csv
│       └── noaug/
│           ├── threshold_sweep_val.csv
│           └── threshold_sweep_test.csv
```
