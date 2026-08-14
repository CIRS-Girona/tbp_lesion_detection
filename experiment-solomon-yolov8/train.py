import os
"""
train.py
--------
Solomon Baseline: YOLOv8s trained with Solomon's original Kaggle settings.

No HP sweep — Solomon did not tune hyperparameters. We train with his default
settings to reproduce his approach as faithfully as possible on our data split.

Two variants:
  --aug_mode full   : default YOLO augmentation (augment=True) -- Solomon's actual setting
  --aug_mode noaug  : horizontal flip only -- for fair ablation comparison

WandB is used for logging so results appear alongside our other models in the
same project dashboard.

Usage:
  python experiment-solomon/train.py --aug_mode full
  python experiment-solomon/train.py --aug_mode noaug

Author: Yamin | June 2026
"""

import argparse
import time
from multiprocessing import freeze_support
from pathlib import Path

import wandb
from ultralytics import YOLO

WANDB_PROJECT = "skin-lesion-detection"
WANDB_ENTITY  = os.environ.get("WANDB_ENTITY", "your-wandb-entity")

SCRIPT_DIR  = Path(__file__).resolve().parent
ITOBOS_ROOT = SCRIPT_DIR.parent.parent
DATA_YAML   = str(ITOBOS_ROOT / "params.yaml")

# Solomon's exact training settings (from yolo_train.py in his GitHub)
MODEL_FILE  = "yolov8s.pt"   # YOLOv8s (README says YOLOv8; we use 's' to match our baseline)
EPOCHS      = 100
IMGSZ       = 1024
BATCH       = 8
SINGLE_CLS  = True
WORKERS     = 4
PATIENCE    = 20             # extended vs Solomon's 7 — consistent with our other experiments
RESULTS_DIR = SCRIPT_DIR / "runs" / "best_model"


def parse_args():
    p = argparse.ArgumentParser(description="Train Solomon baseline (YOLOv8s, default HPs)")
    p.add_argument("--aug_mode", type=str, required=True, choices=["full", "noaug"],
                   help="full = Solomon's default augmentation; noaug = flip only")
    p.add_argument("--device",   type=int, default=0, help="GPU device id (default: 0)")
    p.add_argument("--epochs",   type=int, default=EPOCHS)
    p.add_argument("--patience", type=int, default=PATIENCE)
    return p.parse_args()


def main():
    args     = parse_args()
    run_name = f"solomon_yolov8s_{args.aug_mode}_100ep"

    # Solomon used augment=True (YOLO default aug). We replicate this for full,
    # and use flip-only for noaug (consistent with our other ablation pairs).
    if args.aug_mode == "full":
        aug_kwargs = dict(
            augment=True,           # YOLO default augmentation (Solomon's setting)
            mosaic=1.0,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            flipud=0.0,
            fliplr=0.5,
        )
    else:  # noaug
        aug_kwargs = dict(
            augment=False,
            mosaic=0.0,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            flipud=0.0,
            fliplr=0.5,             # horizontal flip only
        )

    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        tags=["solomon_baseline", "yolov8s", "full_100ep", args.aug_mode,
              "experiment-solomon"],
        config={
            "model":       MODEL_FILE,
            "aug_mode":    args.aug_mode,
            "epochs":      args.epochs,
            "imgsz":       IMGSZ,
            "batch":       BATCH,
            "single_cls":  SINGLE_CLS,
            "optimizer":   "auto",     # Solomon's setting
            "patience":    args.patience,
            "dataset":     DATA_YAML,
            "experiment":  "experiment-solomon",
            "note":        "Solomon baseline — default YOLO HPs, no sweep",
            **aug_kwargs,
        },
    )

    print(f"\n{'='*60}")
    print(f"  Training Solomon Baseline")
    print(f"  Model     : {MODEL_FILE}")
    print(f"  Aug mode  : {args.aug_mode}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Device    : cuda:{args.device}")
    print(f"  iToBoS    : {ITOBOS_ROOT}")
    print(f"{'='*60}\n")

    model   = YOLO(str(ITOBOS_ROOT / MODEL_FILE))
    t_start = time.time()

    results = model.train(
        data=DATA_YAML,
        imgsz=IMGSZ,
        epochs=args.epochs,
        batch=BATCH,
        single_cls=SINGLE_CLS,
        workers=WORKERS,
        device=args.device,
        project=str(RESULTS_DIR),
        name=run_name,
        exist_ok=True,
        plots=True,
        val=True,
        patience=args.patience,
        optimizer="auto",           # Solomon's setting (YOLO default)
        warmup_epochs=3,
        verbose=True,
        **aug_kwargs,
    )

    train_time   = time.time() - t_start
    weights_path = RESULTS_DIR / run_name / "weights" / "best.pt"

    if results and hasattr(results, "results_dict"):
        final = {f"best/{k}": float(v)
                 for k, v in results.results_dict.items()
                 if isinstance(v, (int, float))}
        p_val = final.get("best/metrics/precision(B)", 0.0)
        r_val = final.get("best/metrics/recall(B)",    0.0)
        f1    = 2 * p_val * r_val / (p_val + r_val + 1e-9)
        final["best/F1"]          = round(f1, 6)
        final["train_time_hours"] = round(train_time / 3600, 3)
        wandb.log(final)

        print("\n-- Final Metrics --")
        for k, v in final.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

    save_dir = RESULTS_DIR / run_name
    for fname in ["confusion_matrix.png", "PR_curve.png", "F1_curve.png",
                  "P_curve.png", "R_curve.png", "results.png",
                  "labels.jpg", "labels_correlogram.jpg"]:
        fpath = save_dir / fname
        if fpath.exists():
            wandb.log({fname.replace(".png", "").replace(".jpg", ""): wandb.Image(str(fpath))})

    if weights_path.exists():
        artifact = wandb.Artifact(
            name=f"solomon_yolov8s_{args.aug_mode}",
            type="model",
            metadata={"aug_mode": args.aug_mode, "baseline": "solomon"},
        )
        artifact.add_file(str(weights_path))
        wandb.log_artifact(artifact)

    print(f"\n  Done! Weights at:")
    print(f"    {weights_path}")
    print(f"\nTraining complete in {train_time/3600:.2f} hours")
    wandb.finish()


if __name__ == "__main__":
    freeze_support()
    main()
