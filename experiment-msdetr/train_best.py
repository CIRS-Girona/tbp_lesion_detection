import os
"""
train_best.py
-------------
Full 100-epoch training for MS-DETR using the best HP config from the sweep.

IMPORTANT: Requires the patched ultralytics from yamin/msdetr-src/
           Run inside the MSDETR conda environment.

Usage (from ~/code/iToBoS/):
  conda activate MSDETR
  python yamin/experiment-msdetr/train_best.py --aug_mode full
  python yamin/experiment-msdetr/train_best.py --aug_mode noaug

Author: Yamin | June 2026
"""

import argparse
import json
import time
from multiprocessing import freeze_support
from pathlib import Path

import wandb
from ultralytics import RTDETR

WANDB_PROJECT  = "skin-lesion-detection"
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "your-wandb-entity")

SCRIPT_DIR     = Path(__file__).resolve().parent
ITOBOS_ROOT    = SCRIPT_DIR.parent.parent
DATA_YAML      = str(ITOBOS_ROOT / "params.yaml")
MSDETR_SRC     = SCRIPT_DIR.parent / "msdetr-src"
MODEL_YAML     = str(MSDETR_SRC / "ultralytics" / "cfg" / "models" / "rt-detr" / "MSDETR.yaml")

FULL_EPOCHS    = 100
IMGSZ          = 1024
BATCH          = 4
DEVICE         = 0
SINGLE_CLS     = True
RESULTS_DIR    = SCRIPT_DIR / "runs" / "best_model"

# Fallback defaults if no sweep config found
DEFAULTS = {
    "lr0": 0.0001, "lrf": 0.01, "momentum": 0.937, "weight_decay": 0.0001,
    "warmup_epochs": 3, "warmup_momentum": 0.8,
    "box": 7.5, "cls": 1.5, "dfl": 1.5, "iou": 0.7,
    "dropout": 0.0, "fliplr": 0.5,
    "mosaic": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "scale": 0.0, "translate": 0.0,
}


def parse_args():
    p = argparse.ArgumentParser(description="Train MS-DETR with best sweep HPs")
    p.add_argument("--aug_mode", type=str, required=True, choices=["full", "noaug"])
    p.add_argument("--epochs",   type=int, default=FULL_EPOCHS)
    p.add_argument("--patience", type=int, default=20)
    return p.parse_args()


def load_best_config(aug_mode: str) -> dict:
    config_path = SCRIPT_DIR / f"best_sweep_config_msdetr_{aug_mode}.json"
    if config_path.exists():
        with open(config_path) as f:
            data = json.load(f)
        cfg = data.get("config", {})
        print(f"Loaded best sweep config:")
        print(f"    F1={data.get('f1', '?'):.4f}  Run ID: {data.get('run_id', '?')}")
        print(f"    Source: {config_path}")
        return {**DEFAULTS, **cfg}
    else:
        print(f"[WARN] Config not found: {config_path}")
        print("  Run hparam_sweep.py first. Using default HPs as fallback.")
        return dict(DEFAULTS)


def main():
    args     = parse_args()
    hp       = load_best_config(args.aug_mode)
    run_name = f"msdetr_best_{args.aug_mode}_100ep"

    if args.aug_mode == "full":
        aug_kwargs = dict(
            mosaic=hp.get("mosaic", 0.0),
            hsv_s=hp.get("hsv_s", 0.0),
            hsv_v=hp.get("hsv_v", 0.0),
            scale=hp.get("scale", 0.0),
            translate=hp.get("translate", 0.0),
            fliplr=hp.get("fliplr", 0.5),
            flipud=0.0,
        )
    else:
        aug_kwargs = dict(
            mosaic=0.0, hsv_s=0.0, hsv_v=0.0,
            scale=0.0, translate=0.0,
            fliplr=hp.get("fliplr", 0.5),
            flipud=0.0,
        )

    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        tags=["best_model", "msdetr", "full_100ep", args.aug_mode, "experiment-msdetr"],
        config={
            **hp,
            "model_variant": "msdetr",
            "aug_mode":      args.aug_mode,
            "epochs":        args.epochs,
            "imgsz":         IMGSZ,
            "batch":         BATCH,
            "single_cls":    SINGLE_CLS,
            "dataset":       DATA_YAML,
            "experiment":    "experiment-msdetr",
            "architecture":  "MS-DETR (MDF backbone + DSA neck + DFFB head + RTDETRDecoder)",
        },
    )

    print(f"\n{'='*60}")
    print(f"  Training MS-DETR -- {args.epochs} epochs")
    print(f"  Aug mode  : {args.aug_mode}")
    print(f"  Run name  : {run_name}")
    print(f"  Model YAML: {MODEL_YAML}")
    print(f"  iToBoS    : {ITOBOS_ROOT}")
    print(f"  Device    : cuda:{DEVICE}")
    print(f"  HPs       : lr0={hp['lr0']:.2e}  box={hp['box']:.2f}  cls={hp['cls']:.2f}")
    print(f"{'='*60}\n")

    model   = RTDETR(MODEL_YAML)
    t_start = time.time()

    results = model.train(
        data=DATA_YAML,
        imgsz=IMGSZ,
        epochs=args.epochs,
        batch=BATCH,
        single_cls=SINGLE_CLS,
        workers=4,
        device=DEVICE,
        project=str(RESULTS_DIR),
        name=run_name,
        exist_ok=True,
        plots=True,
        val=True,
        patience=args.patience,
        optimizer="AdamW",
        verbose=True,
        # Best HPs from sweep
        lr0=hp["lr0"],
        lrf=hp["lrf"],
        momentum=hp["momentum"],
        weight_decay=hp["weight_decay"],
        warmup_epochs=hp["warmup_epochs"],
        warmup_momentum=hp["warmup_momentum"],
        box=hp["box"],
        cls=hp["cls"],
        dfl=hp["dfl"],
        dropout=hp["dropout"],
        iou=hp["iou"],
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
                  "P_curve.png", "R_curve.png", "results.png"]:
        fpath = save_dir / fname
        if fpath.exists():
            wandb.log({fname.replace(".png", ""): wandb.Image(str(fpath))})

    if weights_path.exists():
        artifact = wandb.Artifact(
            name=f"msdetr_best_{args.aug_mode}",
            type="model",
            metadata={**hp, "aug_mode": args.aug_mode, "architecture": "MS-DETR"},
        )
        artifact.add_file(str(weights_path))
        wandb.log_artifact(artifact)

    print(f"\n  Done! Weights at:")
    print(f"    {weights_path}")
    print(f"  Next step: bash yamin/experiment-msdetr/run_evaluate_{args.aug_mode}.sh")
    print(f"\nTraining complete in {train_time/3600:.2f} hours")
    wandb.finish()


if __name__ == "__main__":
    freeze_support()
    main()
