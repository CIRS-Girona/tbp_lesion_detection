import os
"""
evaluate.py
-----------
Threshold sweep evaluation for Solomon baseline (YOLOv8s).

Runs the same 11x9 confidence x NMS IoU grid as all other models.
Reports:
  - Default confidence result  (Solomon's original conf=0.325 from test_models.py)
  - Tuned confidence result    (best from sweep -- for fair comparison with our models)

Usage:
  python experiment-solomon/evaluate.py \\
      --weights experiment-solomon/runs/best_model/solomon_yolov8s_full_100ep/weights/best.pt \\
      --name full

  python experiment-solomon/evaluate.py \\
      --weights experiment-solomon/runs/best_model/solomon_yolov8s_noaug_100ep/weights/best.pt \\
      --name noaug

Author: Yamin | June 2026
"""

import argparse
import csv
import time
from pathlib import Path

import wandb
from ultralytics import YOLO

WANDB_PROJECT = "skin-lesion-detection"
WANDB_ENTITY  = os.environ.get("WANDB_ENTITY", "your-wandb-entity")

SCRIPT_DIR  = Path(__file__).resolve().parent
ITOBOS_ROOT = SCRIPT_DIR.parent.parent
DATA_YAML   = str(ITOBOS_ROOT / "params.yaml")

IMGSZ      = 1024
BATCH      = 8
SINGLE_CLS = True

# Same grid as all other experiments
CONF_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
IOU_THRESHOLDS  = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

# Solomon's original confidence (from his test_models.py)
SOLOMON_DEFAULT_CONF = 0.325
SOLOMON_DEFAULT_IOU  = 0.45


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Solomon baseline with threshold sweep")
    p.add_argument("--weights",  type=str, required=True, help="Path to best.pt")
    p.add_argument("--name",     type=str, required=True,
                   choices=["full", "noaug"], help="Condition label")
    p.add_argument("--device",   type=int, default=0)
    return p.parse_args()


def run_validation(model, split: str, conf: float, iou: float,
                   save_dir: Path, log_prefix: str, device: int) -> dict:
    metrics = model.val(
        data=DATA_YAML,
        split=split,
        imgsz=IMGSZ,
        batch=BATCH,
        conf=conf,
        iou=iou,
        device=device,
        single_cls=SINGLE_CLS,
        project=str(save_dir),
        name=f"{split}_conf{conf:.2f}_iou{iou:.2f}",
        exist_ok=True,
        plots=True,
        save_json=True,
        verbose=False,
    )
    result = {}
    if hasattr(metrics, "box"):
        b = metrics.box
        p, r = float(b.mp), float(b.mr)
        result = {
            f"{log_prefix}/mAP50":     float(b.map50),
            f"{log_prefix}/mAP50_95":  float(b.map),
            f"{log_prefix}/precision": p,
            f"{log_prefix}/recall":    r,
            f"{log_prefix}/f1":        2 * p * r / (p + r + 1e-9),
        }
    return result


def threshold_sweep(model, split: str, save_dir: Path, run, device: int) -> list:
    print(f"\n-- Threshold Sweep on [{split}] --")
    rows  = []
    table = wandb.Table(columns=["split", "conf", "iou_nms",
                                 "mAP50", "mAP50_95", "precision", "recall", "f1"])
    for conf in CONF_THRESHOLDS:
        for iou in IOU_THRESHOLDS:
            try:
                metrics = model.val(
                    data=DATA_YAML,
                    split=split,
                    imgsz=IMGSZ,
                    batch=BATCH,
                    conf=conf,
                    iou=iou,
                    device=device,
                    single_cls=SINGLE_CLS,
                    project=str(save_dir / "threshold_sweep"),
                    name=f"{split}_c{conf:.2f}_i{iou:.2f}",
                    exist_ok=True,
                    plots=False,
                    verbose=False,
                )
                if hasattr(metrics, "box"):
                    b = metrics.box
                    p, r, m50, m95 = float(b.mp), float(b.mr), float(b.map50), float(b.map)
                    f1  = 2 * p * r / (p + r + 1e-9)
                    row = dict(split=split, conf=conf, iou_nms=iou,
                               mAP50=m50, mAP50_95=m95, precision=p, recall=r, f1=f1)
                    rows.append(row)
                    table.add_data(split, conf, iou, m50, m95, p, r, f1)
                    print(f"  conf={conf:.2f}  iou={iou:.2f}  -> "
                          f"mAP50={m50:.4f}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}")
            except Exception as e:
                print(f"  [warn] conf={conf} iou={iou} failed: {e}")

    run.log({f"threshold_sweep/{split}": table})

    if rows:
        csv_path = save_dir / f"threshold_sweep_{split}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        artifact = wandb.Artifact(f"solomon_threshold_sweep_{split}", type="analysis")
        artifact.add_file(str(csv_path))
        run.log_artifact(artifact)
        print(f"  Saved: {csv_path}")

    return rows


def find_best_threshold(rows: list) -> tuple:
    if not rows:
        return {}, {}
    return (max(rows, key=lambda r: r["mAP50"]),
            max(rows, key=lambda r: r["f1"]))


def main():
    args     = parse_args()
    weights  = Path(args.weights)
    save_dir = SCRIPT_DIR / "runs" / "evaluation" / args.name

    if not weights.exists():
        print(f"Weights not found: {weights}")
        return

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"solomon_yolov8s_{args.name}_evaluation",
        tags=["evaluation", "solomon_baseline", "yolov8s", "experiment-solomon"],
        config={
            "weights":              str(weights),
            "solomon_default_conf": SOLOMON_DEFAULT_CONF,
            "solomon_default_iou":  SOLOMON_DEFAULT_IOU,
            "imgsz":                IMGSZ,
            "batch":                BATCH,
            "experiment":           "experiment-solomon",
        },
    )

    print(f"\n{'='*60}")
    print(f"  Evaluating Solomon Baseline : {weights}")
    print(f"  Solomon original conf       : {SOLOMON_DEFAULT_CONF}")
    print(f"  iToBoS                      : {ITOBOS_ROOT}")
    print(f"{'='*60}\n")

    model = YOLO(str(weights))

    # 1. Solomon's original confidence (what he submitted to Kaggle)
    print(f"-- Solomon default conf={SOLOMON_DEFAULT_CONF} --")
    for split in ["val", "test"]:
        t0 = time.time()
        m  = run_validation(model, split, SOLOMON_DEFAULT_CONF, SOLOMON_DEFAULT_IOU,
                            save_dir, f"solomon_default/{split}", args.device)
        m[f"solomon_default/{split}/eval_time_s"] = round(time.time() - t0, 2)
        run.log(m)
        print(f"  [{split}]  F1={m.get(f'solomon_default/{split}/f1', 0):.4f}  "
              f"mAP50={m.get(f'solomon_default/{split}/mAP50', 0):.4f}  "
              f"P={m.get(f'solomon_default/{split}/precision', 0):.4f}  "
              f"R={m.get(f'solomon_default/{split}/recall', 0):.4f}")

    # 2. Full threshold sweep
    print("\nStarting full threshold sweep (conf x NMS IoU)...")

    val_rows  = threshold_sweep(model, "val",  save_dir, run, args.device)
    bv_map, bv_f1 = find_best_threshold(val_rows)
    if bv_f1:
        print(f"\n  Best val by F1:    conf={bv_f1['conf']:.2f}  "
              f"iou={bv_f1['iou_nms']:.2f}  F1={bv_f1['f1']:.4f}")
        run.log({
            "best_threshold/val_f1_conf":  bv_f1["conf"],
            "best_threshold/val_f1_iou":   bv_f1["iou_nms"],
            "best_threshold/val_f1_value": bv_f1["f1"],
        })

    test_rows = threshold_sweep(model, "test", save_dir, run, args.device)
    bt_map, bt_f1 = find_best_threshold(test_rows)
    if bt_f1:
        print(f"\n  Best test by F1:   conf={bt_f1['conf']:.2f}  "
              f"iou={bt_f1['iou_nms']:.2f}  F1={bt_f1['f1']:.4f}")
        run.log({
            "best_threshold/test_f1_conf":  bt_f1["conf"],
            "best_threshold/test_f1_iou":   bt_f1["iou_nms"],
            "best_threshold/test_f1_value": bt_f1["f1"],
        })

    # 3. Summary
    print(f"\n{'='*60}")
    print(f"  SOLOMON BASELINE EVALUATION SUMMARY")
    print(f"  Weights: {weights}")
    if bt_f1:
        print(f"\n  Default  conf={SOLOMON_DEFAULT_CONF:.3f} -- see solomon_default/* in WandB")
        print(f"  Tuned    conf={bt_f1['conf']:.2f}  iou={bt_f1['iou_nms']:.2f}  "
              f"F1={bt_f1['f1']:.4f}")
    print(f"{'='*60}\n")

    wandb.finish()


if __name__ == "__main__":
    main()
