import os
"""
evaluate.py
-----------
Comprehensive threshold sweep evaluation for YOLOv11s.

Identical to experiment-v12/evaluate.py but for YOLOv11s.
Runs val + test threshold sweeps (conf & NMS IoU), logs everything to WandB.

Note: YOLOv11s uses standard NMS (NMS-based), same as YOLOv12s.
From our v12s results, NMS IoU has negligible effect for TBP lesion detection
(0.37-0.75 pp F1 spread max). We still sweep it to confirm the same pattern.

Usage:
  # No-aug model:
  python experiment-v11/evaluate.py \\
      --weights experiment-v11/runs/best_model/yolo11s_best_noaug_100ep/weights/best.pt \\
      --run_name yolo11s_noaug_evaluation \\
      --name noaug

  # Full-aug model:
  python experiment-v11/evaluate.py \\
      --weights experiment-v11/runs/best_model/yolo11s_best_full_100ep/weights/best.pt \\
      --run_name yolo11s_full_evaluation \\
      --name full

Author: Yamin | May 2026
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
DEVICE     = 0
SINGLE_CLS = True

# Threshold sweep ranges — same as v12 for direct comparison
CONF_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
IOU_THRESHOLDS  = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate YOLOv11s model with threshold sweep")
    p.add_argument("--weights",  type=str, required=True,
                   help="Path to best.pt weights file")
    p.add_argument("--run_name", type=str, default="yolo11s_evaluation",
                   help="WandB run name")
    p.add_argument("--name",     type=str, default=None,
                   help="Output subfolder name. Use --name noaug or --name full "
                        "to prevent overwriting between runs.")
    p.add_argument("--conf",     type=float, default=0.20,
                   help="Default confidence threshold for initial eval")
    p.add_argument("--iou",      type=float, default=0.50,
                   help="Default NMS IoU threshold for initial eval")
    p.add_argument("--no_threshold_sweep", action="store_true",
                   help="Skip the full threshold sweep (faster, for debugging)")
    return p.parse_args()


def run_validation(model, split: str, conf: float, iou: float,
                   save_dir: Path, log_prefix: str) -> dict:
    """Run model.val() on a split and return metrics dict."""
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
    if metrics and hasattr(metrics, "results_dict"):
        result = {f"{log_prefix}/{k}": float(v)
                  for k, v in metrics.results_dict.items()
                  if isinstance(v, (int, float))}
    if hasattr(metrics, "box"):
        b = metrics.box
        result.update({
            f"{log_prefix}/mAP50":     float(b.map50),
            f"{log_prefix}/mAP50_95":  float(b.map),
            f"{log_prefix}/precision": float(b.mp),
            f"{log_prefix}/recall":    float(b.mr),
        })
    return result


def log_split_plots(wandb_run, results_dir: Path, prefix: str):
    """Log confusion matrix, PR/F1/P/R curves for a split."""
    for fname in ["confusion_matrix.png", "PR_curve.png", "F1_curve.png",
                  "P_curve.png", "R_curve.png"]:
        fpath = results_dir / fname
        if fpath.exists():
            wandb_run.log({f"{prefix}/{fname.replace('.png','')}": wandb.Image(str(fpath))})


def threshold_sweep(model, split: str, save_dir: Path, run) -> list:
    """Sweep conf & NMS IoU thresholds, log comparison table to WandB."""
    print(f"\n-- Threshold Sweep on [{split}] --")
    rows = []
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
        artifact = wandb.Artifact(f"v11_threshold_sweep_{split}", type="analysis")
        artifact.add_file(str(csv_path))
        run.log_artifact(artifact)
        print(f"  Saved: {csv_path}")

    return rows


def find_best_threshold(rows: list) -> tuple:
    """Return (best_by_mAP50, best_by_F1)."""
    if not rows:
        return {}, {}
    return (max(rows, key=lambda r: r["mAP50"]),
            max(rows, key=lambda r: r["f1"]))


def main():
    args    = parse_args()
    weights = Path(args.weights)

    if not weights.exists():
        print(f"Weights not found: {weights}")
        print("  Run train_best.py first, or pass --weights <path>")
        return

    # Use --name to separate results per model (prevents overwriting)
    subfolder = args.name if args.name else args.run_name.replace("/", "_")
    save_dir = SCRIPT_DIR / "runs" / "evaluation" / subfolder

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=args.run_name,
        tags=["evaluation", "yolo11s", "experiment-v11"],
        config={
            "weights":      str(weights),
            "default_conf": args.conf,
            "default_iou":  args.iou,
            "imgsz":        IMGSZ,
            "batch":        BATCH,
            "experiment":   "experiment-v11",
        },
    )

    print(f"\n{'='*60}")
    print(f"  Evaluating : {weights}")
    print(f"  Default    : conf={args.conf}  iou={args.iou}")
    print(f"  iToBoS     : {ITOBOS_ROOT}")
    print(f"{'='*60}\n")

    model = YOLO(str(weights))

    # 1. Val set
    print("-- Validation Set --")
    t0 = time.time()
    val_metrics = run_validation(model, "val", args.conf, args.iou, save_dir, "val")
    val_metrics["val/eval_time_s"] = round(time.time() - t0, 2)
    run.log(val_metrics)
    print(f"  mAP50     = {val_metrics.get('val/mAP50', 0):.4f}")
    print(f"  Precision = {val_metrics.get('val/precision', 0):.4f}")
    print(f"  Recall    = {val_metrics.get('val/recall', 0):.4f}")
    log_split_plots(run, save_dir / f"val_conf{args.conf:.2f}_iou{args.iou:.2f}", "val_plots")

    # 2. Test set
    print("\n-- Test Set --")
    t0 = time.time()
    test_metrics = run_validation(model, "test", args.conf, args.iou, save_dir, "test")
    test_metrics["test/eval_time_s"] = round(time.time() - t0, 2)
    run.log(test_metrics)
    print(f"  mAP50     = {test_metrics.get('test/mAP50', 0):.4f}")
    print(f"  Precision = {test_metrics.get('test/precision', 0):.4f}")
    print(f"  Recall    = {test_metrics.get('test/recall', 0):.4f}")
    log_split_plots(run, save_dir / f"test_conf{args.conf:.2f}_iou{args.iou:.2f}", "test_plots")

    # 3. Comparison table
    run.log({"val_vs_test": wandb.Table(
        columns=["split", "mAP50", "mAP50_95", "precision", "recall"],
        data=[
            ["val",  val_metrics.get("val/mAP50",  0), val_metrics.get("val/mAP50_95",  0),
                     val_metrics.get("val/precision", 0), val_metrics.get("val/recall", 0)],
            ["test", test_metrics.get("test/mAP50", 0), test_metrics.get("test/mAP50_95", 0),
                     test_metrics.get("test/precision", 0), test_metrics.get("test/recall", 0)],
        ],
    )})

    # 4. Threshold sweep
    if not args.no_threshold_sweep:
        print("\nStarting threshold sweep (conf x NMS IoU)...")

        val_rows  = threshold_sweep(model, "val",  save_dir, run)
        bv_map, bv_f1 = find_best_threshold(val_rows)
        if bv_map:
            print(f"\n  Best val by mAP50: conf={bv_map['conf']:.2f}  iou={bv_map['iou_nms']:.2f}  mAP50={bv_map['mAP50']:.4f}")
            print(f"  Best val by F1:    conf={bv_f1['conf']:.2f}  iou={bv_f1['iou_nms']:.2f}  F1={bv_f1['f1']:.4f}")
            run.log({
                "best_threshold/val_map50_conf":  bv_map["conf"],
                "best_threshold/val_map50_iou":   bv_map["iou_nms"],
                "best_threshold/val_map50_value": bv_map["mAP50"],
                "best_threshold/val_f1_conf":     bv_f1["conf"],
                "best_threshold/val_f1_iou":      bv_f1["iou_nms"],
                "best_threshold/val_f1_value":    bv_f1["f1"],
            })

        test_rows = threshold_sweep(model, "test", save_dir, run)
        bt_map, bt_f1 = find_best_threshold(test_rows)
        if bt_map:
            print(f"\n  Best test by mAP50: conf={bt_map['conf']:.2f}  iou={bt_map['iou_nms']:.2f}  mAP50={bt_map['mAP50']:.4f}")
            print(f"  Best test by F1:    conf={bt_f1['conf']:.2f}  iou={bt_f1['iou_nms']:.2f}  F1={bt_f1['f1']:.4f}")
            run.log({
                "best_threshold/test_map50_conf":  bt_map["conf"],
                "best_threshold/test_map50_iou":   bt_map["iou_nms"],
                "best_threshold/test_map50_value": bt_map["mAP50"],
                "best_threshold/test_f1_conf":     bt_f1["conf"],
                "best_threshold/test_f1_iou":      bt_f1["iou_nms"],
                "best_threshold/test_f1_value":    bt_f1["f1"],
            })
    else:
        print("\n  (Threshold sweep skipped)")

    # 5. Model artifact
    artifact = wandb.Artifact("yolo11s_evaluated_model", type="model")
    artifact.add_file(str(weights))
    run.log_artifact(artifact)

    print(f"\nEvaluation complete. WandB run: {run.name}")
    wandb.finish()


if __name__ == "__main__":
    main()
