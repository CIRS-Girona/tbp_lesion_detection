import os
"""
train_best.py
-------------
Full 100-epoch training for YOLOv11s using the best HP config from the sweep.

Usage:
  # No-augmentation best model:
  python experiment-v11/train_best.py --aug_mode noaug

  # Full-augmentation best model:
  python experiment-v11/train_best.py --aug_mode full

Author: Yamin | May 2026
"""

import argparse
import json
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

MODEL_FILE  = "yolo11s.pt"
FULL_EPOCHS = 100
IMGSZ       = 1024
BATCH       = 8
WORKERS     = 4
DEVICE      = 0
SINGLE_CLS  = True
RESULTS_DIR = SCRIPT_DIR / "runs" / "best_model"

# Fallback defaults if no sweep config found
DEFAULTS = {
    "lr0": 0.01, "lrf": 0.01, "weight_decay": 0.0005, "dropout": 0.0,
    "momentum": 0.937, "warmup_epochs": 3, "warmup_momentum": 0.8,
    "box": 7.5, "cls": 0.5, "dfl": 1.5, "iou": 0.7,
    "fliplr": 0.5, "mosaic": 0.0, "mixup": 0.0,
    "hsv_s": 0.0, "hsv_v": 0.0, "translate": 0.0, "scale": 0.0,
}


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv11s with best sweep HPs")
    p.add_argument("--aug_mode", type=str, required=True, choices=["noaug", "full"],
                   help="Which best config to load: noaug or full")
    p.add_argument("--epochs",   type=int, default=FULL_EPOCHS)
    p.add_argument("--patience", type=int, default=20)
    return p.parse_args()


def load_best_config(aug_mode: str) -> dict:
    """Load best HP config from sweep JSON; fall back to defaults if missing."""
    model_tag   = MODEL_FILE.replace(".pt", "")
    config_path = SCRIPT_DIR / f"best_sweep_config_{model_tag}_{aug_mode}.json"

    if config_path.exists():
        with open(config_path) as f:
            data = json.load(f)
        cfg = data.get("config", {})
        print(f"Loaded best sweep config:")
        print(f"    F1={data.get('f1', '?'):.4f}  mAP50={data.get('map50', '?'):.4f}")
        print(f"    Run ID : {data.get('run_id', '?')}")
        print(f"    Source : {config_path}")
        return {**DEFAULTS, **cfg}
    else:
        print(f"Config not found: {config_path}")
        print("  Run hparam_sweep.py first, then re-run this script.")
        print("  Using default hyperparameters as fallback.")
        return dict(DEFAULTS)


def main():
    args      = parse_args()
    model_tag = MODEL_FILE.replace(".pt", "")
    hp        = load_best_config(args.aug_mode)
    run_name  = f"{model_tag}_best_{args.aug_mode}_100ep"

    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        tags=["best_model", model_tag, "full_100ep", args.aug_mode, "experiment-v11"],
        config={
            **hp,
            "model_variant": model_tag,
            "aug_mode":      args.aug_mode,
            "epochs":        args.epochs,
            "imgsz":         IMGSZ,
            "batch":         BATCH,
            "single_cls":    SINGLE_CLS,
            "dataset":       DATA_YAML,
            "experiment":    "experiment-v11",
        },
    )

    print(f"\n{'='*60}")
    print(f"  Training {model_tag} -- {args.epochs} epochs")
    print(f"  Aug mode  : {args.aug_mode}")
    print(f"  Run name  : {run_name}")
    print(f"  iToBoS    : {ITOBOS_ROOT}")
    print(f"  HPs       : {hp}")
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
        device=DEVICE,
        project=str(RESULTS_DIR),
        name=run_name,
        exist_ok=True,
        plots=True,
        val=True,
        patience=args.patience,
        optimizer="AdamW",      # NOT 'auto' -- auto ignores lr0 & momentum
        verbose=True,
        lr0=hp["lr0"],
        lrf=hp["lrf"],
        weight_decay=hp["weight_decay"],
        dropout=hp["dropout"],
        momentum=hp["momentum"],
        warmup_epochs=hp["warmup_epochs"],
        warmup_momentum=hp["warmup_momentum"],
        box=hp["box"],
        cls=hp["cls"],
        dfl=hp["dfl"],
        iou=hp["iou"],
        fliplr=hp["fliplr"],
        mosaic=hp["mosaic"],
        mixup=hp["mixup"],
        hsv_s=hp["hsv_s"],
        hsv_v=hp["hsv_v"],
        translate=hp["translate"],
        scale=hp["scale"],
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
            name=f"{model_tag}_best_{args.aug_mode}",
            type="model",
            metadata={**hp, "aug_mode": args.aug_mode},
        )
        artifact.add_file(str(weights_path))
        wandb.log_artifact(artifact)
        print(f"\nBest weights : {weights_path}")
        print(f"Artifact logged to WandB")

    print(f"\nTraining complete in {train_time/3600:.2f} hours")
    wandb.finish()


if __name__ == "__main__":
    freeze_support()
    main()
