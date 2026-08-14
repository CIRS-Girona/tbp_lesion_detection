import os
"""
evaluate.py
-----------
Threshold sweep evaluation for MS-DETR.

Runs the same 11x9 confidence x NMS IoU grid as all YOLO models.
RT-DETR is technically NMS-free (Hungarian matching at training), but
Ultralytics applies NMS at inference time for compatibility — so the
sweep is still meaningful and keeps comparison fair.

IMPORTANT: Requires the patched ultralytics from yamin/msdetr-src/
           Run inside the MSDETR conda environment.

Usage (from ~/code/iToBoS/):
  conda activate MSDETR
  python yamin/experiment-msdetr/evaluate.py \\
      --weights yamin/experiment-msdetr/runs/best_model/msdetr_best_full_100ep/weights/best.pt \\
      --name full

  python yamin/experiment-msdetr/evaluate.py \\
      --weights yamin/experiment-msdetr/runs/best_model/msdetr_best_noaug_100ep/weights/best.pt \\
      --name noaug

Author: Yamin | June 2026
"""

import argparse
import csv
import time
from pathlib import Path

import wandb
from ultralytics import RTDETR

WANDB_PROJECT   = "skin-lesion-detection"
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "your-wandb-entity")

SCRIPT_DIR      = Path(__file__).resolve().parent
ITOBOS_ROOT     = SCRIPT_DIR.parent.parent
DATA_YAML       = str(ITOBOS_ROOT / "params.yaml")

IMGSZ           = 1024
BATCH           = 4
DEVICE          = 0
SINGLE_CLS      = True

# Identical grid to all other experiments
CONF_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
IOU_THRESHOLDS  = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MS-DETR with threshold sweep")
    p.add_argument("--weights",  type=str, required=True, help="Path to best.pt")
    p.add_argument("--name",     type=str, required=True, choices=["full", "noaug"])
    p.add_argument("--conf",     type=float, default=0.20, help="Initial eval conf")
    p.add_argument("--iou",      type=float, default=0.50, help="Initial eval NMS IoU")
    return p.parse_args()


def run_validation(model, split: str, conf: float, iou: float,
                   save_dir: Path, log_prefix: str) -> dict:
    metrics = model.val(
        data=DATA_YAML,
        split=split,
        imgsz=IMGSZ,
        batch=BATCH,
        conf=conf,
        iou=iou,
        device=DEVICE,
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


def threshold_sweep(model, split: str, save_dir: Path, run) -> list:
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
                    device=DEVICE,
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
        artifact = wandb.Artifact(f"msdetr_threshold_sweep_{split}", type="analysis")
        artifact.add_file(str(csv_path))
        run.log_artifact(artifact)
        print(f"  Saved: {csv_path}")

    return rows


def find_best(rows: list) -> tuple:
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
        name=f"msdetr_{args.name}_evaluation",
        tags=["evaluation", "msdetr", "experiment-msdetr", args.name],
        config={
            "weights":      str(weights),
            "default_conf": args.conf,
            "default_iou":  args.iou,
            "imgsz":        IMGSZ,
            "batch":        BATCH,
            "experiment":   "experiment-msdetr",
            "architecture": "MS-DETR (MDF + DSA + DFFB + RTDETRDecoder)",
        },
    )

    print(f"\n{'='*60}")
    print(f"  Evaluating MS-DETR : {weights}")
    print(f"  Default            : conf={args.conf}  iou={args.iou}")
    print(f"{'='*60}\n")

    model = RTDETR(str(weights))

    # Default conf evaluation (val + test)
    print(f"-- Default eval: conf={args.conf}  iou={args.iou} --")
    for split in ["val", "test"]:
        t0 = time.time()
        m  = run_validation(model, split, args.conf, args.iou, save_dir, split)
        m[f"{split}/eval_time_s"] = round(time.time() - t0, 2)
        run.log(m)
        print(f"  [{split}]  F1={m.get(f'{split}/f1', 0):.4f}  "
              f"mAP50={m.get(f'{split}/mAP50', 0):.4f}  "
              f"P={m.get(f'{split}/precision', 0):.4f}  "
              f"R={m.get(f'{split}/recall', 0):.4f}")

    print("\nStarting full threshold sweep (11 conf x 9 NMS IoU)...")

    val_rows  = threshold_sweep(model, "val",  save_dir, run)
    _, bv_f1  = find_best(val_rows)
    if bv_f1:
        print(f"\n  Best val by F1:   conf={bv_f1['conf']:.2f}  "
              f"iou={bv_f1['iou_nms']:.2f}  F1={bv_f1['f1']:.4f}")
        run.log({"best_threshold/val_f1_conf":  bv_f1["conf"],
                 "best_threshold/val_f1_iou":   bv_f1["iou_nms"],
                 "best_threshold/val_f1_value": bv_f1["f1"]})

    test_rows = threshold_sweep(model, "test", save_dir, run)
    bt_map, bt_f1 = find_best(test_rows)
    if bt_f1:
        print(f"\n  Best test by F1:  conf={bt_f1['conf']:.2f}  "
              f"iou={bt_f1['iou_nms']:.2f}  F1={bt_f1['f1']:.4f}")
        print(f"  Best test by mAP: conf={bt_map['conf']:.2f}  "
              f"iou={bt_map['iou_nms']:.2f}  mAP50={bt_map['mAP50']:.4f}")
        run.log({"best_threshold/test_f1_conf":   bt_f1["conf"],
                 "best_threshold/test_f1_iou":    bt_f1["iou_nms"],
                 "best_threshold/test_f1_value":  bt_f1["f1"],
                 "best_threshold/test_map50_conf": bt_map["conf"],
                 "best_threshold/test_map50_value": bt_map["mAP50"]})

    print(f"\n{'='*60}")
    print(f"  MS-DETR EVALUATION COMPLETE")
    print(f"  Weights: {weights}")
    if bt_f1:
        print(f"  Best test F1 = {bt_f1['f1']:.4f} @ conf={bt_f1['conf']:.2f}")
        print(f"  Best mAP50   = {bt_map['mAP50']:.4f} @ conf={bt_map['conf']:.2f}")
    print(f"{'='*60}\n")

    wandb.finish()


if __name__ == "__main__":
    main()
