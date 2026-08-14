#!/usr/bin/env python3
"""
TIDE Error Analysis for all models.
Ref: Bolya et al., ECCV 2020 — https://dbolya.github.io/tide/

Decomposes detection errors into:
  Cls  - correct box, wrong class (N/A here — single class)
  Loc  - correct class, IoU too low (<0.5)
  Both - wrong class AND bad localisation
  Dupe - correct but duplicate prediction
  Bkg  - FP on background (no GT nearby)
  Miss - FN (GT not detected)

For NMS vs NMS-free analysis: run on YOLOv12s vs YOLOv26s.
For best-model analysis: run on D-FINE and YOLOv12s.

Install: pip install tidecv

Usage:
  python scripts/tide_analysis.py \\
    --gt-json /path/to/dataset_coco/annotations/instances_test.json \\
    --pred-dir /path/to/yolo_preds/ \\
    --dfine-preds /path/to/dfine_raw_preds.json \\
    --output-dir results/tide_analysis/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DATASET_DIR = Path(os.environ.get("DFINE_DATASET_DIR", "~/dfine_results/dataset_coco")).expanduser()
RESULTS_DIR = Path(__file__).resolve().parent / "results"
EVAL_DIR    = RESULTS_DIR / "evaluation"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TIDE error analysis for all models")
    p.add_argument("--split",      choices=["val","test"], default="test")
    p.add_argument("--gt-json",    type=Path, default=None,
                   help="Path to COCO GT annotations JSON (default: auto-resolved)")
    p.add_argument("--output-dir", type=Path,
                   default=RESULTS_DIR / "tide_analysis")
    # Per-model prediction JSONs in COCO format
    p.add_argument("--dfine-fullaug",  type=Path, default=None,
                   help="D-FINE full-aug raw_preds.json")
    p.add_argument("--dfine-noaug",    type=Path, default=None,
                   help="D-FINE no-aug raw_preds.json")
    p.add_argument("--yolov12-preds",  type=Path, default=None,
                   help="YOLOv12s COCO predictions JSON")
    p.add_argument("--yolov26-preds",  type=Path, default=None,
                   help="YOLOv26s COCO predictions JSON (NMS-free)")
    p.add_argument("--yolov8-preds",   type=Path, default=None,
                   help="YOLOv8s COCO predictions JSON")
    p.add_argument("--msdetr-preds",   type=Path, default=None,
                   help="MS-DETR COCO predictions JSON")
    p.add_argument("--rtdetr-preds",   type=Path, default=None,
                   help="RT-DETR-L COCO predictions JSON")
    return p.parse_args()


def yolo_to_coco_preds(pred_path: Path, category_id: int = 1) -> list:
    """
    Convert YOLO-format predictions to COCO prediction format.
    YOLO ultralytics saves results as CSV or JSON with xyxy boxes.
    This handles both formats.
    """
    data = json.loads(pred_path.read_text())

    # If already COCO format [{image_id, category_id, bbox, score}]
    if isinstance(data, list) and len(data) > 0 and "bbox" in data[0]:
        return data

    # YOLO results.json format
    coco_preds = []
    for item in data:
        image_id = item.get("image_id", item.get("id"))
        for box in item.get("boxes", []):
            x1, y1, x2, y2 = box["xyxy"]
            coco_preds.append({
                "image_id":   int(image_id),
                "category_id": category_id,
                "bbox":        [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                "score":       float(box.get("conf", box.get("score", 1.0))),
            })
    return coco_preds


def run_tide_analysis(gt_path: Path, pred_json: Path, model_name: str,
                      out_dir: Path) -> dict:
    """Run TIDE analysis for a single model."""
    try:
        from tidecv import TIDE, datasets  # type: ignore
    except ImportError:
        print("TIDE not installed. Run: pip install tidecv")
        return {}

    import matplotlib
    matplotlib.use("Agg")

    out_dir.mkdir(parents=True, exist_ok=True)

    preds_raw = json.loads(pred_json.read_text())
    if isinstance(preds_raw, list) and len(preds_raw) > 0 and "bbox" in preds_raw[0]:
        preds = preds_raw
    else:
        preds = yolo_to_coco_preds(pred_json)

    # Force prediction category_id and image_id to match ground truth
    try:
        gt_data = json.loads(gt_path.read_text())

        # Align category ID
        target_cat_id = 1
        if "categories" in gt_data and gt_data["categories"]:
            target_cat_id = gt_data["categories"][0]["id"]
        elif "annotations" in gt_data and gt_data["annotations"]:
            target_cat_id = gt_data["annotations"][0]["category_id"]

        # Align image ID (maps YOLO string IDs like 'image_0158' to integer IDs)
        name_to_id = {}
        for img in gt_data.get("images", []):
            img_id = img["id"]
            fname = img["file_name"]
            stem = Path(fname).stem
            name_to_id[str(img_id)] = img_id
            name_to_id[stem] = img_id
            name_to_id[fname] = img_id
            if "_" in stem:
                parts = stem.split("_")
                for i in range(len(parts) - 1):
                    if parts[i] == "image":
                        name_to_id[f"image_{parts[i+1]}"] = img_id

        unmapped = 0
        for p in preds:
            p["category_id"] = target_cat_id
            pid = str(p["image_id"])
            if pid in name_to_id:
                p["image_id"] = name_to_id[pid]
            else:
                unmapped += 1

        if unmapped > 0:
            print(f"  [warn] Could not map {unmapped}/{len(preds)} prediction image IDs to GT!")
    except Exception as e:
        print(f"  [warn] Alignment failed: {e}")

    # Save to temp file for TIDE
    tmp_pred_path = out_dir / f"{model_name.replace(' ','_')}_coco_preds.json"
    tmp_pred_path.write_text(json.dumps(preds))

    tide = TIDE()
    gt   = datasets.COCO(str(gt_path))
    pr   = datasets.COCOResult(str(tmp_pred_path))

    run_obj = tide.evaluate(gt, pr, mode=TIDE.BOX)
    tide.summarize()

    tide.plot(str(out_dir / f"tide_{model_name.replace(' ','_')}"))

    # Extract error breakdown
    results = {}
    main_errors = run_obj.fix_main_errors()
    for error_class, val in main_errors.items():
        name = error_class.short_name if hasattr(error_class, "short_name") else error_class.__name__.lower().replace("error", "")
        results[name] = round(float(val), 4)

    (out_dir / f"tide_{model_name.replace(' ','_')}_summary.json").write_text(
        json.dumps({"model": model_name, "errors": results}, indent=2)
    )

    print(f"\n{'='*50}")
    print(f"  TIDE: {model_name}")
    print(f"{'='*50}")
    for k, v in results.items():
        print(f"  {k:20s}: {v:.4f}")

    return results


def compare_nms_vs_nmsfree(gt_path: Path, yolov12_preds: Path, yolov26_preds: Path,
                             out_dir: Path) -> None:
    """
    Additional NMS vs NMS-free analysis:
    - Number of redundant predictions (IoU > 0.5 between two detections)
    - Score gap between 1st and 2nd best predictions per image
    This is for YOLOv12 (NMS) vs YOLOv26 (NMS-free) comparison.
    """
    import numpy as np
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    def load_preds_by_image(path: Path) -> dict:
        data = json.loads(path.read_text())
        by_img: dict = {}
        for p in data:
            by_img.setdefault(p["image_id"], []).append(p)
        # Sort by score descending
        for k in by_img:
            by_img[k].sort(key=lambda x: x["score"], reverse=True)
        return by_img

    def count_redundant(by_img: dict, iou_thr: float = 0.5) -> dict:
        """Count boxes per image with IoU > iou_thr with a higher-scored box."""
        redundant_counts = []
        score_gaps = []
        for img_id, preds in by_img.items():
            redundant = 0
            boxes = []
            for p in preds:
                x, y, w, h = p["bbox"]
                boxes.append([x, y, x+w, y+h])
            boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0,4))
            # For each box, check IoU with all higher-scored boxes
            for i in range(1, len(boxes)):
                for j in range(i):
                    b1, b2 = boxes[j], boxes[i]
                    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
                    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
                    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
                    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
                    iou = inter / (a1 + a2 - inter + 1e-9)
                    if iou > iou_thr:
                        redundant += 1
                        break
            redundant_counts.append(redundant)
            # Score gap
            if len(preds) >= 2:
                score_gaps.append(preds[0]["score"] - preds[1]["score"])
        return {"redundant_per_image": redundant_counts, "score_gaps": score_gaps}

    v12_by_img = load_preds_by_image(yolov12_preds)
    v26_by_img = load_preds_by_image(yolov26_preds)
    v12_stats  = count_redundant(v12_by_img)
    v26_stats  = count_redundant(v26_by_img)

    summary = {
        "YOLOv12s (NMS)": {
            "mean_redundant_per_image": float(np.mean(v12_stats["redundant_per_image"])),
            "mean_score_gap":           float(np.mean(v12_stats["score_gaps"])) if v12_stats["score_gaps"] else 0,
        },
        "YOLOv26s (NMS-free)": {
            "mean_redundant_per_image": float(np.mean(v26_stats["redundant_per_image"])),
            "mean_score_gap":           float(np.mean(v26_stats["score_gaps"])) if v26_stats["score_gaps"] else 0,
        },
    }
    (out_dir / "nms_vs_nmsfree_summary.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(v12_stats["redundant_per_image"], bins=20, alpha=0.6, label="YOLOv12s (NMS)", color="steelblue")
    axes[0].hist(v26_stats["redundant_per_image"], bins=20, alpha=0.6, label="YOLOv26s (NMS-free)", color="tomato")
    axes[0].set_xlabel("Redundant boxes per image (IoU>0.5 with higher-scored box)")
    axes[0].set_ylabel("Image count")
    axes[0].set_title("Redundant predictions: NMS vs NMS-free")
    axes[0].legend()
    if v12_stats["score_gaps"] and v26_stats["score_gaps"]:
        axes[1].hist(v12_stats["score_gaps"], bins=30, alpha=0.6, label="YOLOv12s (NMS)", color="steelblue")
        axes[1].hist(v26_stats["score_gaps"], bins=30, alpha=0.6, label="YOLOv26s (NMS-free)", color="tomato")
        axes[1].set_xlabel("Score gap (1st best − 2nd best prediction)")
        axes[1].set_ylabel("Image count")
        axes[1].set_title("Score gap: top-1 vs top-2 prediction")
        axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_dir / "nms_vs_nmsfree_analysis.png", dpi=200)
    plt.close()
    print(f"\nSaved NMS analysis: {out_dir}/nms_vs_nmsfree_analysis.png")
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()

    gt_json = args.gt_json
    if gt_json is None:
        gt_json = (DATASET_DIR / "annotations" / f"instances_{args.split}.json").resolve()

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load GT JSON and ensure 'segmentation' field exists (TIDE requires it even for bbox-only datasets)
    print("Loading and preparing GT annotations for TIDE compatibility...")
    gt_data = json.loads(gt_json.read_text())
    has_seg = True
    if "annotations" in gt_data:
        for ann in gt_data["annotations"]:
            if "segmentation" not in ann or not ann["segmentation"]:
                if "bbox" in ann:
                    x, y, w, h = ann["bbox"]
                    ann["segmentation"] = [[x, y, x + w, y, x + w, y + h, x, y + h]]
                else:
                    ann["segmentation"] = []
                has_seg = False

    if not has_seg:
        gt_tide_path = out_dir / "gt_tide_with_segmentation.json"
        gt_tide_path.write_text(json.dumps(gt_data))
        print(f"Created TIDE-compatible GT annotation file at: {gt_tide_path}")
        gt_json = gt_tide_path

    all_results: dict = {}

    models = [
        ("D-FINE Full-Aug",  args.dfine_fullaug),
        ("D-FINE No-Aug",    args.dfine_noaug),
        ("YOLOv12s",         args.yolov12_preds),
        ("YOLOv26s",         args.yolov26_preds),
        ("YOLOv8s",          args.yolov8_preds),
        ("MS-DETR",          args.msdetr_preds),
        ("RT-DETR-L",        args.rtdetr_preds),
    ]

    for model_name, pred_path in models:
        if pred_path is None:
            print(f"[skip] {model_name}: no predictions path provided")
            continue
        pred_path = pred_path.expanduser().resolve()
        if not pred_path.exists():
            print(f"[skip] {model_name}: file not found: {pred_path}")
            continue
        results = run_tide_analysis(gt_json, pred_path, model_name, out_dir)
        if results:
            all_results[model_name] = results

    # NMS vs NMS-free detailed analysis
    if args.yolov12_preds and args.yolov26_preds:
        if args.yolov12_preds.exists() and args.yolov26_preds.exists():
            print("\n=== NMS vs NMS-free redundancy analysis ===")
            compare_nms_vs_nmsfree(
                gt_json,
                args.yolov12_preds.expanduser().resolve(),
                args.yolov26_preds.expanduser().resolve(),
                out_dir,
            )

    if all_results:
        print("\n" + "="*70)
        print("TIDE Error Summary (AP impact)")
        print("="*70)
        error_types = sorted(set(k for v in all_results.values() for k in v.keys()))
        header = f"{'Model':<22} " + " ".join(f"{e:8s}" for e in error_types)
        print(header)
        print("-"*70)
        for model_name, errs in all_results.items():
            row = f"{model_name:<22} " + " ".join(f"{errs.get(e, 0):8.4f}" for e in error_types)
            print(row)

        (out_dir / "tide_summary_all.json").write_text(
            json.dumps(all_results, indent=2)
        )
        print(f"\nFull results: {out_dir}/tide_summary_all.json")


if __name__ == "__main__":
    main()
