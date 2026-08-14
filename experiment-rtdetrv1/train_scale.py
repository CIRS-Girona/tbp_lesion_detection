import os
"""
train_scale.py
--------------
Train the SCALE-enhanced RT-DETR-L on iToBoS.

Uses the SAME default hyperparameters as the baseline (train_best.py defaults)
so the comparison isolates the architecture change — NOT a re-tuned model.
This is the scientifically fair setup for an architecture-contribution paper.

Ablation: toggle individual SCALE branches with --no_dsa / --no_far / --no_sare.
  Full model:  python train_scale.py --aug_mode full --device 2
  Ablation:    python train_scale.py --aug_mode full --device 2 --no_far --no_sare

Author: Praveen Kumar Murali | June 2026
"""

import argparse
import time
from multiprocessing import freeze_support
from pathlib import Path

import wandb

import scale_rtdetr  # must be imported and enabled BEFORE building RTDETR

WANDB_PROJECT = "skin-lesion-detection"
WANDB_ENTITY  = os.environ.get("WANDB_ENTITY", "your-wandb-entity")

SCRIPT_DIR  = Path(__file__).resolve().parent
ITOBOS_ROOT = SCRIPT_DIR.parent.parent
DATA_YAML   = str(ITOBOS_ROOT / "params.yaml")

MODEL_FILE  = "rtdetr-l.pt"
FULL_EPOCHS = 100
IMGSZ       = 1024
BATCH       = 8
WORKERS     = 4
SINGLE_CLS  = True
RESULTS_DIR = SCRIPT_DIR / "runs" / "scale_model"

# Same default HPs the baseline used (keep identical for a fair comparison)
DEFAULT_HP = dict(
    lr0=0.0001, lrf=0.1, momentum=0.9, weight_decay=0.0001,
    warmup_epochs=2, warmup_momentum=0.8,
    fliplr=0.5, mosaic=0.5, hsv_s=0.3, hsv_v=0.4, scale=0.3, translate=0.1,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train SCALE-enhanced RT-DETR-L")
    p.add_argument("--aug_mode", type=str, required=True, choices=["noaug", "full"])
    p.add_argument("--epochs",   type=int, default=FULL_EPOCHS)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--device",   type=int, default=0)
    # Ablation toggles (default: all three branches ON = full SCALE)
    p.add_argument("--no_dsa",  action="store_true", help="disable Dynamic Scale Aggregation")
    p.add_argument("--no_far",  action="store_true", help="disable Frequency-Aware Reconstruction")
    p.add_argument("--no_sare", action="store_true", help="disable Scale-Aware Residual Enhancement")
    p.add_argument("--suffix", type=str, default="", help="extra run-name suffix, e.g. g1")
    p.add_argument("--deep_type", type=str, default="cf", choices=["cf", "fft"],
                   help="deepest-scale module: cf=paper-faithful Context Fusion, fft=our FFT module")
    return p.parse_args()


def main():
    args = parse_args()

    use_dsa  = not args.no_dsa
    use_far  = not args.no_far
    use_sare = not args.no_sare

    # Enable the SCALE patch BEFORE building the model.
    scale_rtdetr.enable_scale(use_dsa=use_dsa, use_far=use_far, use_sare=use_sare,
                              deep_type=args.deep_type)
    from ultralytics import RTDETR  # imported after patch is in place

    # Build an ablation tag for the run name
    tag = "scale"
    if not (use_dsa and use_far and use_sare):
        parts = []
        if use_dsa:  parts.append("dsa")
        if use_far:  parts.append("far")
        if use_sare: parts.append("sare")
        tag = "scale_" + ("_".join(parts) if parts else "none")

    hp = dict(DEFAULT_HP)
    if args.aug_mode == "noaug":
        hp.update(mosaic=0.0, hsv_s=0.0, hsv_v=0.0, scale=0.0, translate=0.0)

    run_name = f"rtdetr-l_{tag}_{args.aug_mode}_100ep" + (f"_{args.suffix}" if args.suffix else "")

    wandb.init(
        project=WANDB_PROJECT, entity=WANDB_ENTITY, name=run_name,
        tags=["scale_model", "rtdetr-l", args.aug_mode, tag, "experiment-rtdetr"],
        config={**hp, "use_dsa": use_dsa, "use_far": use_far, "use_sare": use_sare,
                "aug_mode": args.aug_mode, "epochs": args.epochs, "imgsz": IMGSZ,
                "batch": BATCH, "dataset": DATA_YAML, "experiment": "experiment-rtdetr"},
    )

    print(f"\n{'='*60}")
    print(f"  SCALE-RT-DETR training  ({tag})")
    print(f"  Branches  : dsa={use_dsa} far={use_far} sare={use_sare}")
    print(f"  Aug mode  : {args.aug_mode}   Run: {run_name}")
    print(f"  HPs       : {hp}")
    print(f"{'='*60}\n")

    # Build from YAML so the patched __init__ runs (loading .pt bypasses it),
    # then load pretrained weights (SCALE params stay freshly initialised).
    model = RTDETR("rtdetr-l.yaml")
    model.load(MODEL_FILE)
    t_start = time.time()

    results = model.train(
        data=DATA_YAML, imgsz=IMGSZ, epochs=args.epochs, batch=BATCH,
        single_cls=SINGLE_CLS, workers=WORKERS, device=args.device,
        project=str(RESULTS_DIR), name=run_name, exist_ok=True,
        plots=True, val=True, patience=args.patience, optimizer="AdamW", verbose=True,
        lr0=hp["lr0"], lrf=hp["lrf"], weight_decay=hp["weight_decay"],
        momentum=hp["momentum"], warmup_epochs=hp["warmup_epochs"],
        warmup_momentum=hp["warmup_momentum"],
        fliplr=hp["fliplr"], mosaic=hp["mosaic"], hsv_s=hp["hsv_s"],
        hsv_v=hp["hsv_v"], translate=hp["translate"], scale=hp["scale"],
    )

    train_time = time.time() - t_start
    save_dir = RESULTS_DIR / run_name

    if results and hasattr(results, "results_dict"):
        final = {f"best/{k}": float(v) for k, v in results.results_dict.items()
                 if isinstance(v, (int, float))}
        p_val = final.get("best/metrics/precision(B)", 0.0)
        r_val = final.get("best/metrics/recall(B)", 0.0)
        f1 = 2 * p_val * r_val / (p_val + r_val + 1e-9)
        final["best/F1"] = round(f1, 6)
        final["train_time_hours"] = round(train_time / 3600, 3)
        wandb.log(final)
        print("\n── Final Metrics ──")
        for k, v in final.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

    for fname in ["confusion_matrix.png", "BoxPR_curve.png", "BoxF1_curve.png",
                  "results.png", "labels.jpg"]:
        fpath = save_dir / fname
        if fpath.exists():
            wandb.log({fname.split('.')[0]: wandb.Image(str(fpath))})

    weights_path = save_dir / "weights" / "best.pt"
    if weights_path.exists():
        art = wandb.Artifact(name=f"rtdetr-l_{tag}_{args.aug_mode}", type="model",
                             metadata={**hp, "use_dsa": use_dsa, "use_far": use_far,
                                       "use_sare": use_sare})
        art.add_file(str(weights_path))
        wandb.log_artifact(art)
        print(f"\nBest weights : {weights_path}")

    print(f"\nTraining complete in {train_time/3600:.2f} hours")
    wandb.finish()


if __name__ == "__main__":
    freeze_support()
    main()
