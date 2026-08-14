import os
"""
evaluate.py
-----------
Confidence threshold sweep evaluation for RT-DETR-L on iToBoS.

RT-DETR is NMS-FREE (end-to-end Hungarian matching, no NMS post-processing).
Therefore the NMS IoU threshold has no effect at inference. To save compute
we fix iou=0.50 and sweep only the confidence threshold. (Set --iou_sweep to
also sweep a few IoU values and confirm the results are flat, demonstrating
the NMS-free property for the paper.)

Reports F1/P/R at each conf, and mAP50 at conf=0.001 (full PR curve) so the
result matches the paper's metric convention.

Usage:
  python evaluate.py \
      --weights runs/best_model/rtdetr-l_best_full_100ep/weights/best.pt \
      --run_name rtdetr-l_full_evaluation --name full

Author: Praveen Kumar Murali | June 2026
"""

import argparse
import csv
import time
from pathlib import Path

import wandb
from ultralytics import RTDETR

WANDB_PROJECT = "skin-lesion-detection"
WANDB_ENTITY  = os.environ.get("WANDB_ENTITY", "your-wandb-entity")

SCRIPT_DIR  = Path(__file__).resolve().parent
ITOBOS_ROOT = SCRIPT_DIR.parent.parent
DATA_YAML   = str(ITOBOS_ROOT / "params.yaml")

IMGSZ      = 1024
BATCH      = 8
DEVICE     = 0
SINGLE_CLS = True

# RT-DETR outputs low confidences, so its optimal F1 threshold
# is far below YOLO's — include low values (down to 0.001).
CONF_THRESHOLDS = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20,
                   0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
# RT-DETR is NMS-free: single IoU is enough. --iou_sweep expands this.
IOU_THRESHOLDS_SINGLE = [0.50]
IOU_THRESHOLDS_DEMO   = [0.30, 0.50, 0.70]   # to demonstrate flatness if requested
# mAP50 reported at near-zero conf to approximate the full PR curve (paper rule).
MAP_CONF = 0.001


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RT-DETR-L with confidence sweep")
    p.add_argument("--weights",  type=str, required=True,
                   help="Path to best.pt weights file")
    p.add_argument("--run_name", type=str, default="rtdetr-l_evaluation",
                   help="WandB run name")
    p.add_argument("--name",     type=str, default=None,
                   help="Output subfolder name (separates noaug/full results)")
    p.add_argument("--conf",     type=float, default=0.20,
                   help="Default confidence threshold for initial eval")
    p.add_argument("--iou",      type=float, default=0.50,
                   help="IoU threshold (irrelevant for NMS-free RT-DETR)")
    p.add_argument("--device",   type=int, default=DEVICE, help="GPU index (default: 0)")
    p.add_argument("--iou_sweep", action="store_true",
                   help="Also sweep a few IoU values to demonstrate NMS-free flatness")
    p.add_argument("--no_threshold_sweep", action="store_true",
                   help="Skip the full threshold sweep (faster, for debugging)")
    return p.parse_args()


def run_validation(model, split: str, conf: float, iou: float,
                   save_dir: Path, log_prefix: str, device: int) -> dict:
    """Run model.val() on a split and return metrics dict."""
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
        name=f"{split}_conf{conf:.3f}_iou{iou:.2f}",
        exist_ok=True,
        plots=True,
        save_json=True,
        verbose=False,
    )

    result = {}
    if hasattr(metrics, "box"):
        b = metrics.box
        p, r = float(b.mp), float(b.mr)
        result.update({
            f"{log_prefix}/mAP50":     float(b.map50),
            f"{log_prefix}/mAP50_95":  float(b.map),
            f"{log_prefix}/precision": p,
            f"{log_prefix}/recall":    r,
            f"{log_prefix}/f1":        2 * p * r / (p + r + 1e-9),
        })
    return result


def log_split_plots(wandb_run, results_dir: Path, prefix: str):
    """Log confusion matrix, PR/F1/P/R curves for a split."""
    for fname in ["confusion_matrix.png", "PR_curve.png", "F1_curve.png",
                  "P_curve.png", "R_curve.png"]:
        fpath = results_dir / fname
        if fpath.exists():
            wandb_run.log({f"{prefix}/{fname.replace('.png','')}": wandb.Image(str(fpath))})


def threshold_sweep(model, split: str, iou_values: list, save_dir: Path,
                    run, device: int) -> list:
    """Sweep confidence (and optionally IoU) thresholds, log table to WandB."""
    print(f"\n── Threshold Sweep on [{split}] ──")
    rows = []
    table = wandb.Table(columns=["split", "conf", "iou_nms",
                                 "mAP50", "mAP50_95", "precision", "recall", "f1"])

    for conf in CONF_THRESHOLDS:
        for iou in iou_values:
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
        artifact = wandb.Artifact(f"rtdetr_threshold_sweep_{split}", type="analysis")
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

    subfolder  = args.name if args.name else args.run_name.replace("/", "_")
    save_dir   = SCRIPT_DIR / "runs" / "evaluation" / subfolder
    iou_values = IOU_THRESHOLDS_DEMO if args.iou_sweep else IOU_THRESHOLDS_SINGLE

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=args.run_name,
        tags=["evaluation", "rtdetr-l", "experiment-rtdetr"],
        config={
            "weights":      str(weights),
            "default_conf": args.conf,
            "default_iou":  args.iou,
            "imgsz":        IMGSZ,
            "batch":        BATCH,
            "nms_free":     True,
            "experiment":   "experiment-rtdetr",
        },
    )

    print(f"\n{'='*60}")
    print(f"  Evaluating : {weights}")
    print(f"  Default    : conf={args.conf}  (RT-DETR is NMS-free)")
    print(f"{'='*60}\n")

    model = RTDETR(str(weights))

    # mAP50 at conf=0.001 (full PR curve — paper convention)
    print("── mAP50 at conf=0.001 (full PR curve) ──")
    map_val  = run_validation(model, "val",  MAP_CONF, args.iou, save_dir, "val_map",  args.device)
    map_test = run_validation(model, "test", MAP_CONF, args.iou, save_dir, "test_map", args.device)
    run.log({**map_val, **map_test})
    print(f"  val  mAP50 = {map_val.get('val_map/mAP50', 0):.4f}")
    print(f"  test mAP50 = {map_test.get('test_map/mAP50', 0):.4f}")

    print("\n── Validation Set (conf={:.2f}) ──".format(args.conf))
    t0 = time.time()
    val_metrics = run_validation(model, "val", args.conf, args.iou, save_dir, "val", args.device)
    val_metrics["val/eval_time_s"] = round(time.time() - t0, 2)
    run.log(val_metrics)
    print(f"  F1        = {val_metrics.get('val/f1', 0):.4f}")
    print(f"  Precision = {val_metrics.get('val/precision', 0):.4f}")
    print(f"  Recall    = {val_metrics.get('val/recall', 0):.4f}")
    log_split_plots(run, save_dir / f"val_conf{args.conf:.3f}_iou{args.iou:.2f}", "val_plots")

    print("\n── Test Set (conf={:.2f}) ──".format(args.conf))
    t0 = time.time()
    test_metrics = run_validation(model, "test", args.conf, args.iou, save_dir, "test", args.device)
    test_metrics["test/eval_time_s"] = round(time.time() - t0, 2)
    run.log(test_metrics)
    print(f"  F1        = {test_metrics.get('test/f1', 0):.4f}")
    print(f"  Precision = {test_metrics.get('test/precision', 0):.4f}")
    print(f"  Recall    = {test_metrics.get('test/recall', 0):.4f}")
    log_split_plots(run, save_dir / f"test_conf{args.conf:.3f}_iou{args.iou:.2f}", "test_plots")

    if not args.no_threshold_sweep:
        print("\n Starting confidence threshold sweep...")

        val_rows = threshold_sweep(model, "val", iou_values, save_dir, run, args.device)
        bv_map, bv_f1 = find_best_threshold(val_rows)
        if bv_f1:
            print(f"\n  Best val by F1:    conf={bv_f1['conf']:.2f}  F1={bv_f1['f1']:.4f}")
            run.log({
                "best_threshold/val_f1_conf":  bv_f1["conf"],
                "best_threshold/val_f1_value": bv_f1["f1"],
            })

        test_rows = threshold_sweep(model, "test", iou_values, save_dir, run, args.device)
        bt_map, bt_f1 = find_best_threshold(test_rows)
        if bt_f1:
            print(f"\n  Best test by F1:    conf={bt_f1['conf']:.2f}  F1={bt_f1['f1']:.4f}")
            print(f"  Best test by mAP50: conf={bt_map['conf']:.2f}  mAP50={bt_map['mAP50']:.4f}")
            run.log({
                "best_threshold/test_f1_conf":     bt_f1["conf"],
                "best_threshold/test_f1_value":    bt_f1["f1"],
                "best_threshold/test_map50_value": bt_map["mAP50"],
            })
    else:
        print("\n  (Threshold sweep skipped)")

    artifact = wandb.Artifact("rtdetr-l_evaluated_model", type="model")
    artifact.add_file(str(weights))
    run.log_artifact(artifact)

    print(f"\nEvaluation complete. WandB run: {run.name}")
    wandb.finish()


if __name__ == "__main__":
    main()
