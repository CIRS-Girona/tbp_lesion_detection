import os
"""
quick_train.py
--------------
Quick 20-epoch feasibility run for RT-DETR-L on the iToBoS skin lesion dataset.

RT-DETR differences vs YOLO (important for understanding results):
  - End-to-end detection: no NMS post-processing (like YOLOv26s)
  - Uses Hungarian bipartite matching for training loss (not IoU-based assignment)
  - Uses ResNet-50 backbone + transformer encoder-decoder
  - ~32M params (larger than YOLOv11/12/26s ~9-10M)
  - Slower training per epoch than YOLO
  - No DFL loss — uses GIoU + L1 box regression loss

Goal: verify the model trains on iToBoS, estimate performance,
and decide feasibility for the full pipeline.

Usage:
  python quick_train.py
  python quick_train.py --epochs 40 --device 0

Author: Praveen Kumar Murali | June 2026
"""

import argparse
import time
from pathlib import Path

import wandb
from ultralytics import RTDETR

WANDB_PROJECT = "skin-lesion-detection"
WANDB_ENTITY  = os.environ.get("WANDB_ENTITY", "your-wandb-entity")

SCRIPT_DIR  = Path(__file__).resolve().parent
ITOBOS_ROOT = SCRIPT_DIR.parent.parent
DATA_YAML   = str(ITOBOS_ROOT / "params.yaml")

MODEL_FILE  = "rtdetr-l.pt"   # auto-downloads if not present (~65 MB)
RESULTS_DIR = SCRIPT_DIR / "runs" / "quick_train"

IMGSZ      = 1024
BATCH      = 8
WORKERS    = 4
SINGLE_CLS = True

# RT-DETR uses much lower lr than YOLO (transformer-based models need smaller LR)
DEFAULT_HP = dict(
    lr0=0.0001,
    lrf=0.1,
    momentum=0.9,
    weight_decay=0.0001,
    warmup_epochs=2,
    warmup_momentum=0.8,
    box=5.0,
    cls=0.5,
    fliplr=0.5,
    mosaic=0.5,
    hsv_s=0.3,
    hsv_v=0.4,
    scale=0.3,
    translate=0.1,
)


def parse_args():
    p = argparse.ArgumentParser(description="Quick RT-DETR feasibility training on iToBoS")
    p.add_argument("--epochs",   type=int, default=20,
                   help="Number of epochs (default: 20 for quick feasibility check)")
    p.add_argument("--device",   type=int, default=0,
                   help="GPU index (default: 0)")
    p.add_argument("--batch",    type=int, default=BATCH)
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience (default: 10)")
    return p.parse_args()


def main():
    args = parse_args()

    run_name = f"rtdetr-l_quick_{args.epochs}ep"

    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        tags=["rtdetr", "rtdetr-l", "quick_train", "feasibility", "praveen"],
        config={
            **DEFAULT_HP,
            "model":      MODEL_FILE,
            "epochs":     args.epochs,
            "imgsz":      IMGSZ,
            "batch":      args.batch,
            "single_cls": SINGLE_CLS,
            "dataset":    DATA_YAML,
            "experiment": "experiment-rtdetr",
            "note":       "feasibility check — 20ep quick run",
        },
    )

    print(f"\n{'='*62}")
    print(f"  iToBoS | RT-DETR-L Quick Training")
    print(f"  Epochs  : {args.epochs}  |  Batch: {args.batch}  |  GPU: {args.device}")
    print(f"  Data    : {DATA_YAML}")
    print(f"  Output  : {RESULTS_DIR / run_name}")
    print(f"  WandB   : {WANDB_PROJECT} / {run_name}")
    print(f"{'='*62}\n")

    model   = RTDETR(MODEL_FILE)
    t_start = time.time()

    results = model.train(
        data=DATA_YAML,
        imgsz=IMGSZ,
        epochs=args.epochs,
        batch=args.batch,
        single_cls=SINGLE_CLS,
        workers=WORKERS,
        device=args.device,
        project=str(RESULTS_DIR),
        name=run_name,
        exist_ok=True,
        plots=True,
        val=True,
        patience=args.patience,
        optimizer="AdamW",
        verbose=True,
        lr0=DEFAULT_HP["lr0"],
        lrf=DEFAULT_HP["lrf"],
        momentum=DEFAULT_HP["momentum"],
        weight_decay=DEFAULT_HP["weight_decay"],
        warmup_epochs=DEFAULT_HP["warmup_epochs"],
        warmup_momentum=DEFAULT_HP["warmup_momentum"],
        box=DEFAULT_HP["box"],
        cls=DEFAULT_HP["cls"],
        fliplr=DEFAULT_HP["fliplr"],
        mosaic=DEFAULT_HP["mosaic"],
        hsv_s=DEFAULT_HP["hsv_s"],
        hsv_v=DEFAULT_HP["hsv_v"],
        scale=DEFAULT_HP["scale"],
        translate=DEFAULT_HP["translate"],
    )

    train_time = time.time() - t_start

    metrics_dict = {}
    if results and hasattr(results, "results_dict"):
        metrics_dict = {k: float(v) for k, v in results.results_dict.items()
                        if isinstance(v, (int, float))}

    precision = metrics_dict.get("metrics/precision(B)", 0.0)
    recall    = metrics_dict.get("metrics/recall(B)",    0.0)
    map50     = metrics_dict.get("metrics/mAP50(B)",     0.0)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    summary = {
        "final/F1":           round(f1, 6),
        "final/precision":    round(precision, 6),
        "final/recall":       round(recall, 6),
        "final/mAP50":        round(map50, 6),
        "train_time_minutes": round(train_time / 60, 2),
    }
    wandb.log(summary)

    save_dir = RESULTS_DIR / run_name
    for fname in ["confusion_matrix.png", "PR_curve.png", "F1_curve.png",
                  "P_curve.png", "R_curve.png", "results.png",
                  "labels.jpg", "labels_correlogram.jpg"]:
        fpath = save_dir / fname
        if fpath.exists():
            wandb.log({fname.replace(".png", "").replace(".jpg", ""): wandb.Image(str(fpath))})

    weights_path = save_dir / "weights" / "best.pt"

    print(f"\n{'='*62}")
    print(f"  RESULTS — RT-DETR-L Quick Training ({args.epochs} epochs)")
    print(f"  F1        : {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  mAP50     : {map50:.4f}")
    print(f"  Time      : {train_time/60:.1f} min  ({train_time/3600:.2f} hrs)")
    print(f"  Weights   : {weights_path}")
    print(f"{'='*62}")
    print(f"\n  Reference (Phase 1 best at 100ep): YOLOv12s Full F1=0.640")
    print(f"  Compare your {args.epochs}-ep RT-DETR result against this.\n")

    wandb.finish()


if __name__ == "__main__":
    main()
