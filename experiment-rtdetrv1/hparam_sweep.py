import os
"""
hparam_sweep.py
---------------
WandB Bayesian Hyperparameter Sweep -- RT-DETR-L Skin Lesion Detection

RT-DETR's loss in Ultralytics uses fixed internal gains (class=1, bbox=5,
giou=2). The box/cls/dfl arguments do NOT affect RT-DETR training, so they
are NOT tuned here. Instead we tune the params that actually matter for a
transformer detector: learning rate, regularisation, optimiser momentum,
warmup, dropout, and augmentation.

RT-DETR is also NMS-free (end-to-end), so there is no IoU/NMS param to tune.

Handles both aug modes via --aug_mode flag:
  --aug_mode noaug : only horizontal flip
  --aug_mode full  : standard augmentation (mosaic, hsv, scale, translate)

How it works:
  1. wandb.agent() calls this script repeatedly (once per HP trial).
  2. HP config is passed via wandb.config.
  3. Each run trains RT-DETR-L for SWEEP_EPOCHS (40) on iToBoS (40 trials total).
  4. Best config (by F1) saved to best_sweep_config_rtdetr-l_{aug_mode}.json
     for use by train_best.py.

Author: Praveen Kumar Murali | June 2026
"""

import time
import json
import argparse
from pathlib import Path
from multiprocessing import freeze_support

import wandb
from ultralytics import RTDETR

WANDB_PROJECT = "skin-lesion-detection"
WANDB_ENTITY  = os.environ.get("WANDB_ENTITY", "your-wandb-entity")

SCRIPT_DIR  = Path(__file__).resolve().parent
ITOBOS_ROOT = SCRIPT_DIR.parent.parent
DATA_YAML   = str(ITOBOS_ROOT / "params.yaml")

SWEEP_EPOCHS = 25   # shorter trials; train_best.py runs the full 100 ep
IMGSZ        = 1024
BATCH        = 8
WORKERS      = 4
DEVICE       = 0
SINGLE_CLS   = True


def parse_args():
    p = argparse.ArgumentParser(description="WandB HP sweep for RT-DETR-L")
    p.add_argument("--model",    type=str, default="rtdetr-l.pt",
                   help="Model weights filename (auto-downloads if missing)")
    p.add_argument("--aug_mode", type=str, default="full",
                   choices=["noaug", "full"],
                   help="noaug: only fliplr | full: standard augmentation")
    p.add_argument("--device",   type=int, default=0,
                   help="GPU device index (default: 0)")
    return p.parse_args()


def train_sweep(args):
    """Called by wandb.agent() for each HP trial."""
    model_tag    = args.model.replace(".pt", "")
    results_dir  = SCRIPT_DIR / "runs" / "sweeps" / f"{model_tag}_{args.aug_mode}"
    best_cfg_out = SCRIPT_DIR / f"best_sweep_config_{model_tag}_{args.aug_mode}.json"

    run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)
    cfg = wandb.config

    run_name = f"sweep_{model_tag}_{args.aug_mode}_{run.id}"

    print(f"\n{'='*60}")
    print(f"  Model     : {args.model}  |  Aug mode: {args.aug_mode}")
    print(f"  GPU device: {args.device}")
    print(f"  Trial ID  : {run.id}")
    print(f"  Config    : {dict(cfg)}")
    print(f"{'='*60}\n")

    wandb.config.update({
        "model_variant": model_tag,
        "aug_mode":      args.aug_mode,
        "sweep_epochs":  SWEEP_EPOCHS,
        "imgsz":         IMGSZ,
        "batch":         BATCH,
        "experiment":    "experiment-rtdetr",
    }, allow_val_change=True)

    model = RTDETR(args.model)
    t_start = time.time()

    if args.aug_mode == "noaug":
        aug_kwargs = dict(
            fliplr=cfg.get("fliplr", 0.5),
            mosaic=0.0, mixup=0.0,
            hsv_s=0.0, hsv_v=0.0,
            scale=0.0, translate=0.0,
        )
    else:  # full -- let sweep tune each augmentation parameter
        aug_kwargs = dict(
            fliplr=cfg.get("fliplr", 0.5),
            mosaic=cfg.get("mosaic", 0.5),
            hsv_s=cfg.get("hsv_s", 0.3),
            hsv_v=cfg.get("hsv_v", 0.4),
            scale=cfg.get("scale", 0.3),
            translate=cfg.get("translate", 0.1),
        )

    results = model.train(
        data=DATA_YAML,
        imgsz=IMGSZ,
        epochs=SWEEP_EPOCHS,
        batch=BATCH,
        single_cls=SINGLE_CLS,
        workers=WORKERS,
        device=args.device,
        project=str(results_dir),
        name=run_name,
        exist_ok=True,
        plots=True,
        val=True,
        lr0=cfg.get("lr0", 0.0001),
        lrf=cfg.get("lrf", 0.1),
        weight_decay=cfg.get("weight_decay", 0.0001),
        dropout=cfg.get("dropout", 0.0),
        momentum=cfg.get("momentum", 0.9),
        warmup_epochs=cfg.get("warmup_epochs", 2),
        warmup_momentum=cfg.get("warmup_momentum", 0.8),
        **aug_kwargs,
        patience=10,
        optimizer="AdamW",     # NOT 'auto' -- auto ignores lr0 & momentum
        verbose=False,
    )

    train_time = time.time() - t_start

    metrics_dict = {}
    if results and hasattr(results, "results_dict"):
        metrics_dict = {k: float(v) for k, v in results.results_dict.items()
                        if isinstance(v, (int, float))}

    precision = metrics_dict.get("metrics/precision(B)", 0.0)
    recall    = metrics_dict.get("metrics/recall(B)",    0.0)
    map50     = metrics_dict.get("metrics/mAP50(B)",     0.0)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    summary = {
        "final/F1":           round(f1, 6),
        "final/mAP50":        round(map50, 6),
        "final/mAP50_95":     metrics_dict.get("metrics/mAP50-95(B)", 0.0),
        "final/precision":    round(precision, 6),
        "final/recall":       round(recall, 6),
        "train_time_minutes": round(train_time / 60, 2),
    }
    wandb.log(summary)

    # Log plots
    save_dir = results_dir / run_name
    for fname in ["PR_curve.png", "F1_curve.png", "P_curve.png",
                  "R_curve.png", "results.png"]:
        fpath = save_dir / fname
        if fpath.exists():
            wandb.log({fname.replace(".png", ""): wandb.Image(str(fpath))})

    _maybe_save_best_config(cfg, f1, map50, run.id, best_cfg_out)

    print(f"\n  Trial {run.id} done -- F1={f1:.4f}  mAP50={map50:.4f}  "
          f"P={precision:.4f}  R={recall:.4f}  ({train_time/60:.1f} min)")
    wandb.finish()
    return summary


def _maybe_save_best_config(cfg, f1: float, map50: float, run_id: str, best_cfg_out: Path):
    """Persist the best sweep config by F1 score."""
    best = {}
    if best_cfg_out.exists():
        try:
            with open(best_cfg_out) as f:
                best = json.load(f)
        except Exception:
            pass
    if f1 > best.get("f1", 0.0):
        best = {
            "f1":     f1,
            "map50":  map50,
            "run_id": run_id,
            "config": dict(cfg),
        }
        with open(best_cfg_out, "w") as f:
            json.dump(best, f, indent=2)
        print(f"  New best config saved (F1={f1:.4f}) -> {best_cfg_out}")


if __name__ == "__main__":
    freeze_support()
    train_sweep(parse_args())
