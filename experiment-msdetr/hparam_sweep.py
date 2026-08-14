import os
"""
hparam_sweep.py
---------------
WandB Bayesian Hyperparameter Sweep -- MS-DETR Skin Lesion Detection

MS-DETR is a modified RT-DETR (Ultralytics RTDETR class) with:
  - MDF backbone blocks (Fourier-domain + multi-branch depthwise conv)
  - DSA neck (Dynamic Sparse Attention)
  - DFFB head blocks (frequency-domain feature fusion)
  - RTDETRDecoder (Hungarian matching, 4 detection levels: P2-P5)

IMPORTANT: Requires the patched ultralytics from yamin/msdetr-src/
           Run inside the MSDETR conda environment.

Usage (from ~/code/iToBoS/):
  conda activate MSDETR
  # Full augmentation sweep:
  wandb agent <entity>/<project>/<sweep_id>  (sweep_id from run_msdetr.sh)

  # Or run directly for testing:
  python yamin/experiment-msdetr/hparam_sweep.py --aug_mode full
  python yamin/experiment-msdetr/hparam_sweep.py --aug_mode noaug

Author: Yamin | June 2026
"""

import time
import json
import argparse
from pathlib import Path
from multiprocessing import freeze_support

import wandb
from ultralytics import RTDETR

WANDB_PROJECT  = "skin-lesion-detection"
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "your-wandb-entity")

SCRIPT_DIR     = Path(__file__).resolve().parent
ITOBOS_ROOT    = SCRIPT_DIR.parent.parent
DATA_YAML      = str(ITOBOS_ROOT / "params.yaml")
MSDETR_SRC     = SCRIPT_DIR.parent / "msdetr-src"
MODEL_YAML     = str(MSDETR_SRC / "ultralytics" / "cfg" / "models" / "rt-detr" / "MSDETR.yaml")

SWEEP_EPOCHS   = 30    # fewer than YOLO (40) — DETR trains ~2x slower/epoch
IMGSZ          = 1024
BATCH          = 4     # conservative for DETR transformer decoder at 1024px
DEVICE         = 0
SINGLE_CLS     = True


# Full-aug sweep: LR + loss weights + augmentation params
SWEEP_CONFIG_FULL = {
    "method": "bayes",
    "metric": {"name": "sweep/val_f1", "goal": "maximize"},
    "parameters": {
        # Learning rate
        "lr0":             {"distribution": "log_uniform_values", "min": 1e-5, "max": 5e-3},
        "lrf":             {"distribution": "log_uniform_values", "min": 0.005, "max": 0.15},
        "momentum":        {"distribution": "uniform",            "min": 0.85,  "max": 0.97},
        "weight_decay":    {"distribution": "log_uniform_values", "min": 1e-6,  "max": 1e-2},
        # Warmup
        "warmup_epochs":   {"distribution": "int_uniform",        "min": 2,     "max": 5},
        "warmup_momentum": {"distribution": "uniform",            "min": 0.60,  "max": 0.90},
        # DETR loss weights
        "box":             {"distribution": "uniform",            "min": 3.0,   "max": 10.0},
        "cls":             {"distribution": "uniform",            "min": 0.3,   "max": 3.0},
        "dfl":             {"distribution": "uniform",            "min": 0.8,   "max": 2.0},
        # Regularisation
        "dropout":         {"distribution": "uniform",            "min": 0.0,   "max": 0.30},
        # Augmentation (full aug only)
        "fliplr":          {"distribution": "uniform",            "min": 0.3,   "max": 0.7},
        "hsv_s":           {"distribution": "uniform",            "min": 0.0,   "max": 0.9},
        "hsv_v":           {"distribution": "uniform",            "min": 0.0,   "max": 0.5},
        "mosaic":          {"distribution": "uniform",            "min": 0.0,   "max": 0.5},
        "scale":           {"distribution": "uniform",            "min": 0.1,   "max": 0.6},
        "translate":       {"distribution": "uniform",            "min": 0.0,   "max": 0.15},
        "iou":             {"distribution": "uniform",            "min": 0.50,  "max": 0.80},
    },
}

# No-aug sweep: only LR + loss weights + flip
SWEEP_CONFIG_NOAUG = {
    "method": "bayes",
    "metric": {"name": "sweep/val_f1", "goal": "maximize"},
    "parameters": {
        "lr0":             {"distribution": "log_uniform_values", "min": 1e-5,  "max": 5e-3},
        "lrf":             {"distribution": "log_uniform_values", "min": 0.005, "max": 0.15},
        "momentum":        {"distribution": "uniform",            "min": 0.85,  "max": 0.97},
        "weight_decay":    {"distribution": "log_uniform_values", "min": 1e-6,  "max": 1e-2},
        "warmup_epochs":   {"distribution": "int_uniform",        "min": 2,     "max": 5},
        "warmup_momentum": {"distribution": "uniform",            "min": 0.60,  "max": 0.90},
        "box":             {"distribution": "uniform",            "min": 3.0,   "max": 10.0},
        "cls":             {"distribution": "uniform",            "min": 0.3,   "max": 3.0},
        "dfl":             {"distribution": "uniform",            "min": 0.8,   "max": 2.0},
        "dropout":         {"distribution": "uniform",            "min": 0.0,   "max": 0.30},
        "fliplr":          {"distribution": "uniform",            "min": 0.3,   "max": 0.7},
        "iou":             {"distribution": "uniform",            "min": 0.50,  "max": 0.80},
    },
}

NUM_TRIALS = 15  # fewer than YOLO (25) — DETR is slower per trial


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--aug_mode", type=str, required=True, choices=["full", "noaug"])
    return p.parse_args()


def train_one_trial(aug_mode: str):
    """Called once per WandB agent trial."""
    run = wandb.init()
    hp  = wandb.config

    model_tag = "msdetr"
    run_name  = f"{model_tag}_{aug_mode}_sweep_{run.id}"

    if aug_mode == "full":
        aug_kwargs = dict(
            mosaic=float(hp.get("mosaic", 0.0)),
            hsv_s=float(hp.get("hsv_s", 0.0)),
            hsv_v=float(hp.get("hsv_v", 0.0)),
            scale=float(hp.get("scale", 0.0)),
            translate=float(hp.get("translate", 0.0)),
            fliplr=float(hp.get("fliplr", 0.5)),
            flipud=0.0,
        )
    else:
        aug_kwargs = dict(
            mosaic=0.0, hsv_s=0.0, hsv_v=0.0,
            scale=0.0, translate=0.0,
            fliplr=float(hp.get("fliplr", 0.5)),
            flipud=0.0,
        )

    save_dir = SCRIPT_DIR / "runs" / "sweep" / aug_mode
    model    = RTDETR(MODEL_YAML)

    # Ultralytics registers its own WandB callback (wb.py) that calls
    # wandb.finish() when training ends. This closes OUR outer sweep run,
    # making all subsequent run.log() calls fail with "not initialized".
    # By removing these callbacks, only our sweep code controls wandb state.
    model.callbacks = {
        event: [cb for cb in cbs
                if "wb" not in getattr(cb, "__module__", "")]
        for event, cbs in model.callbacks.items()
    }

    t_start  = time.time()

    # Wrap model.train() to survive the WandB on_train_end callback crash
    # (wb.py:85 np.interp fails with "too small depth" on RTDETR -- non-fatal).
    # Training completes successfully; we extract metrics from model.trainer.
    results = None
    try:
        results = model.train(
            data=DATA_YAML,
            imgsz=IMGSZ,
            epochs=SWEEP_EPOCHS,
            batch=BATCH,
            single_cls=SINGLE_CLS,
            workers=4,
            device=DEVICE,
            project=str(save_dir),
            name=run_name,
            exist_ok=True,
            plots=False,
            val=True,
            patience=10,
            optimizer="AdamW",
            verbose=False,
            # HPs from sweep
            lr0=float(hp.lr0),
            lrf=float(hp.lrf),
            momentum=float(hp.momentum),
            weight_decay=float(hp.weight_decay),
            warmup_epochs=int(hp.warmup_epochs),
            warmup_momentum=float(hp.warmup_momentum),
            box=float(hp.box),
            cls=float(hp.cls),
            dfl=float(hp.dfl),
            dropout=float(hp.dropout),
            iou=float(hp.iou),
            **aug_kwargs,
        )
    except Exception as e:
        # Training completed -- error is only in the on_train_end WandB callback.
        if "too small depth" in str(e) or "interp" in str(e).lower():
            print(f"  [INFO] Suppressed known WandB callback error: {e}")
        else:
            print(f"  [WARN] model.train() raised: {e}")

    # Extract metrics: prefer results object, fall back to model.trainer
    p_val, r_val = 0.0, 0.0
    if results is not None and hasattr(results, "results_dict"):
        p_val = results.results_dict.get("metrics/precision(B)", 0.0)
        r_val = results.results_dict.get("metrics/recall(B)",    0.0)
    elif hasattr(model, "trainer") and model.trainer is not None:
        # Fallback: read from trainer directly (available even after callback crash)
        t = model.trainer
        if hasattr(t, "metrics") and t.metrics:
            p_val = t.metrics.get("metrics/precision(B)", 0.0)
            r_val = t.metrics.get("metrics/recall(B)",    0.0)
        elif hasattr(t, "validator") and t.validator is not None:
            v = t.validator
            if hasattr(v, "metrics") and hasattr(v.metrics, "box"):
                p_val = float(v.metrics.box.mp)
                r_val = float(v.metrics.box.mr)

    val_f1 = 2 * p_val * r_val / (p_val + r_val + 1e-9) if (p_val + r_val) > 0 else 0.0
    print(f"  Trial result: P={p_val:.4f}  R={r_val:.4f}  F1={val_f1:.4f}")

    # Save best config before any wandb calls that might fail
    best_path = SCRIPT_DIR / f"best_sweep_config_msdetr_{aug_mode}.json"
    new_entry = {
        "f1":     val_f1,
        "run_id": run.id,
        "config": {
            "lr0": float(hp.lr0), "lrf": float(hp.lrf),
            "momentum": float(hp.momentum), "weight_decay": float(hp.weight_decay),
            "warmup_epochs": int(hp.warmup_epochs), "warmup_momentum": float(hp.warmup_momentum),
            "box": float(hp.box), "cls": float(hp.cls), "dfl": float(hp.dfl),
            "dropout": float(hp.dropout), "iou": float(hp.iou),
            "fliplr": float(hp.get("fliplr", 0.5)),
            **({k: float(hp.get(k, 0)) for k in ["mosaic","hsv_s","hsv_v","scale","translate"]}
               if aug_mode == "full" else {}),
            "aug_mode": aug_mode, "sweep_epochs": SWEEP_EPOCHS,
            "imgsz": IMGSZ, "batch": BATCH, "experiment": "experiment-msdetr",
        },
    }
    save_as_best = True
    if best_path.exists():
        try:
            with open(best_path) as f:
                existing = json.load(f)
            if existing.get("f1", 0) >= val_f1:
                save_as_best = False
        except Exception:
            pass  # corrupted file — overwrite
    if save_as_best:
        with open(best_path, "w") as f:
            json.dump(new_entry, f, indent=2)
        print(f"  [NEW BEST] F1={val_f1:.4f}  run_id={run.id}  saved -> {best_path}")

    # Log to WandB using saved run object (not global wandb).
    # Ultralytics' internal WandB callback closes the global wandb state after
    # training. Using run.log() (our saved run object) is safe regardless.
    try:
        run.log({
            "sweep/val_f1":      val_f1,
            "sweep/precision":   p_val,
            "sweep/recall":      r_val,
            "sweep/trial_time":  round((time.time() - t_start) / 3600, 3),
        })
    except Exception as e:
        print(f"  [INFO] WandB log skipped (ultralytics closed the run): {e}")
    try:
        run.finish()
    except Exception:
        pass


def main():
    args = parse_args()

    sweep_config = SWEEP_CONFIG_FULL if args.aug_mode == "full" else SWEEP_CONFIG_NOAUG
    sweep_config["name"] = f"msdetr_{args.aug_mode}_sweep"

    # Add fixed fields to all trials
    sweep_config.setdefault("parameters", {}).update({
        "model_variant": {"value": "msdetr"},
        "aug_mode":      {"value": args.aug_mode},
        "sweep_epochs":  {"value": SWEEP_EPOCHS},
        "imgsz":         {"value": IMGSZ},
        "batch":         {"value": BATCH},
        "experiment":    {"value": "experiment-msdetr"},
    })

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
    )
    print(f"\nSweep ID: {sweep_id}")
    print(f"Sweep URL: https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/sweeps/{sweep_id}")

    wandb.agent(
        sweep_id,
        function=lambda: train_one_trial(args.aug_mode),
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        count=NUM_TRIALS,
    )
    print(f"\nSweep complete. Best config saved to:")
    print(f"  {SCRIPT_DIR}/best_sweep_config_msdetr_{args.aug_mode}.json")


if __name__ == "__main__":
    freeze_support()
    main()
