#!/usr/bin/env python3
"""
Evaluate D-FINE (and optionally YOLO models) at multiple MAP IoU thresholds.

Outputs:
  map_multi_threshold_<split>.json    AP at IoU=0.50,0.55,...,0.95 + specific thresholds
  map_multi_threshold_<split>.csv     Table: model x IoU threshold
  map_bar_chart_<split>.png           Bar chart comparing models at each IoU

Usage:
  # D-FINE only:
  python scripts/eval_map_multi_threshold.py \\
    --dfine-config results/configs/dfine_s_fullaug_sweep_100ep.yml \\
    --dfine-weights results/training/full/dfine_s_fullaug_sweep_100ep/best_stg2.pth \\
    --split test

  # With YOLO comparison (pass paths to saved COCO-format predictions JSONs):
  python scripts/eval_map_multi_threshold.py \\
    --dfine-config results/configs/dfine_s_fullaug_sweep_100ep.yml \\
    --dfine-weights results/training/full/dfine_s_fullaug_sweep_100ep/best_stg2.pth \\
    --yolo-preds-dir /path/to/yolo_coco_predictions/ \\
    --split test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

DFINE_REPO  = Path(os.environ.get("DFINE_REPO", "~/code/iToBoS/aritra/dfine_code/D-FINE")).expanduser()
DATASET_DIR = Path(os.environ.get("DFINE_DATASET_DIR", "~/dfine_results/dataset_coco")).expanduser()
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# IoU thresholds to evaluate at
IOU_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
HIGHLIGHT_IOUS = [0.50, 0.70, 0.75, 0.80, 0.90]  # COCO standard + specific highlights


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dfine-repo",     type=Path, default=DFINE_REPO)
    p.add_argument("--dataset-dir",    type=Path, default=DATASET_DIR)
    p.add_argument("--output-dir",     type=Path, default=RESULTS_DIR / "multi_map_eval")
    p.add_argument("--dfine-config",   type=Path, required=True)
    p.add_argument("--dfine-weights",  type=Path, required=True)
    p.add_argument("--split",          choices=["val", "test"], default="test")
    p.add_argument("--img-size",       type=int, default=1024)
    p.add_argument("--device",         type=str, default="cuda:0")
    p.add_argument("--min-conf",       type=float, default=0.001)
    # Optional: YOLO/other model predictions as COCO-format JSON for comparison
    p.add_argument("--extra-preds",    type=str, default=None,
                   help="JSON of extra model predictions for comparison. "
                        "Format: {'ModelName': 'path/to/coco_preds.json', ...}")
    return p.parse_args()


# COCO-style AP computation

def load_gt(ann_file: Path) -> Tuple[List[dict], Dict[int, list]]:
    data = json.loads(ann_file.read_text())
    images = data["images"]
    gt: Dict[int, list] = {int(img["id"]): [] for img in images}
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        x, y, w, h = ann["bbox"]
        gt.setdefault(int(ann["image_id"]), []).append(
            np.array([x, y, x + w, y + h], dtype=np.float32)
        )
    return images, gt


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros(0, dtype=np.float32)
    xx1 = np.maximum(box[0], boxes[:, 0]); yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2]); yy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    a1 = max(0, (box[2]-box[0])) * max(0, (box[3]-box[1]))
    a2 = np.maximum(0, boxes[:,2]-boxes[:,0]) * np.maximum(0, boxes[:,3]-boxes[:,1])
    return inter / (a1 + a2 - inter + 1e-9)


def compute_ap_at_iou(gt_by_image: Dict[int, list], preds: list, iou_thr: float) -> float:
    """Compute AP at a single IoU threshold using 101-point interpolation."""
    if not preds:
        return 0.0
    total_gt = sum(len(v) for v in gt_by_image.values())
    if total_gt == 0:
        return 0.0

    sorted_preds = sorted(preds, key=lambda p: p["score"], reverse=True)
    matched: Dict[int, set] = {k: set() for k in gt_by_image}
    tp = np.zeros(len(sorted_preds), dtype=np.float32)
    fp = np.zeros(len(sorted_preds), dtype=np.float32)

    for i, pred in enumerate(sorted_preds):
        img_id = int(pred["image_id"])
        gts = gt_by_image.get(img_id, [])
        if not gts:
            fp[i] = 1; continue
        boxes = np.stack(gts).astype(np.float32)
        x, y, w, h = pred["bbox"]
        pred_box = np.array([x, y, x+w, y+h], dtype=np.float32)
        ious = box_iou_one_to_many(pred_box, boxes)
        best_idx = int(np.argmax(ious))
        if ious[best_idx] >= iou_thr and best_idx not in matched[img_id]:
            tp[i] = 1; matched[img_id].add(best_idx)
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp); fp_cum = np.cumsum(fp)
    recall = tp_cum / (total_gt + 1e-9)
    precision = tp_cum / (tp_cum + fp_cum + 1e-9)

    ap = 0.0
    for r in np.linspace(0, 1, 101):
        p = precision[recall >= r].max() if np.any(recall >= r) else 0.0
        ap += p / 101.0
    return float(ap)


def compute_ap_at_conf(gt_by_image: Dict[int, list], preds: list,
                       conf_thr: float, iou_thr: float) -> float:
    filtered = [p for p in preds if p["score"] >= conf_thr]
    return compute_ap_at_iou(gt_by_image, filtered, iou_thr)


def eval_all_thresholds(gt_by_image: Dict[int, list], preds: list,
                        conf_thr: float = 0.001) -> Dict[str, float]:
    """Compute AP at all IoU thresholds and return results dict."""
    filtered = [p for p in preds if p["score"] >= conf_thr]
    results = {}
    for iou in IOU_THRESHOLDS:
        ap = compute_ap_at_iou(gt_by_image, filtered, iou)
        results[f"AP@{int(iou*100)}"] = round(ap * 100, 2)

    # COCO mAP (mean over 0.50:0.05:0.95)
    results["mAP@50:95"] = round(
        np.mean([results[f"AP@{int(t*100)}"] for t in IOU_THRESHOLDS]), 2
    )
    return results


# D-FINE inference

def run_dfine_inference(args: argparse.Namespace, images: List[dict],
                        ann_file: Path) -> list:
    """Run D-FINE and return COCO-format predictions."""
    sys.path.insert(0, str(args.dfine_repo.expanduser().resolve()))
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from src.core import YAMLConfig  # type: ignore

    cfg_path = args.dfine_config.expanduser().resolve()
    wt_path  = args.dfine_weights.expanduser().resolve()

    cfg = YAMLConfig(str(cfg_path), resume=str(wt_path))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    ckpt = torch.load(str(wt_path), map_location="cpu")
    if "ema" in ckpt and isinstance(ckpt["ema"], dict) and "module" in ckpt["ema"]:
        state = ckpt["ema"]["module"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    cfg.model.load_state_dict(state, strict=False)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.post  = cfg.postprocessor.deploy()
        def forward(self, imgs, sizes):
            return self.post(self.model(imgs), sizes)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = Model().to(device).eval()
    transform = T.Compose([T.Resize((args.img_size, args.img_size)), T.ToTensor()])
    image_folder = args.dataset_dir.expanduser().resolve() / "images" / args.split

    preds = []
    with torch.no_grad():
        for idx, img_info in enumerate(images, 1):
            img = Image.open(image_folder / img_info["file_name"]).convert("RGB")
            w, h = img.size
            tensor = transform(img).unsqueeze(0).to(device)
            orig_size = torch.tensor([[w, h]], device=device)
            labels, boxes, scores = model(tensor, orig_size)
            labels = labels[0].cpu().numpy()
            boxes  = boxes[0].cpu().numpy()
            scores = scores[0].cpu().numpy()
            for lab, box, score in zip(labels, boxes, scores):
                if score < args.min_conf:
                    continue
                x1, y1, x2, y2 = [max(0.0, float(v)) for v in box]
                preds.append({
                    "image_id": int(img_info["id"]),
                    "category_id": int(lab),
                    "bbox": [x1, y1, max(0, x2-x1), max(0, y2-y1)],
                    "score": float(score),
                })
            if idx % 50 == 0:
                print(f"  Inferred {idx}/{len(images)} images")
    return preds


# Plotting

def plot_comparison(all_results: Dict[str, Dict[str, float]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    iou_labels = [f"AP@{int(t*100)}" for t in IOU_THRESHOLDS] + ["mAP@50:95"]
    x = np.arange(len(iou_labels))
    width = 0.8 / max(len(all_results), 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (model_name, results) in enumerate(all_results.items()):
        vals = [results.get(k, 0) for k in iou_labels]
        bars = ax.bar(x + i * width, vals, width, label=model_name, alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width * (len(all_results)-1) / 2)
    ax.set_xticklabels(iou_labels, rotation=30, ha="right")
    ax.set_ylabel("AP (%)")
    ax.set_title("Multi-threshold MAP comparison\n(AP@50 to AP@95, D-FINE vs baselines)")
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot: {out_path}")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    ann_file    = dataset_dir / "annotations" / f"instances_{args.split}.json"
    out_dir     = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ground truth from {ann_file}")
    images, gt_by_image = load_gt(ann_file)

    all_results: Dict[str, Dict[str, float]] = {}

    print("\n=== Running D-FINE inference ===")
    dfine_preds = run_dfine_inference(args, images, ann_file)
    (out_dir / "dfine_raw_preds.json").write_text(json.dumps(dfine_preds, indent=2))
    print(f"D-FINE predictions: {len(dfine_preds)}")

    for conf in [0.001, 0.50]:
        label = f"D-FINE-S (conf≥{conf})"
        results = eval_all_thresholds(gt_by_image, dfine_preds, conf_thr=conf)
        all_results[label] = results
        print(f"\n{label}:")
        for k, v in results.items():
            print(f"  {k}: {v:.2f}%")

    if args.extra_preds:
        extra = json.loads(args.extra_preds)
        for model_name, pred_path in extra.items():
            print(f"\n=== Loading {model_name} predictions ===")
            preds = json.loads(Path(pred_path).expanduser().read_text())
            results = eval_all_thresholds(gt_by_image, preds, conf_thr=0.001)
            all_results[model_name] = results
            print(f"{model_name}:")
            for k, v in results.items():
                print(f"  {k}: {v:.2f}%")

    summary_path = out_dir / f"map_multi_threshold_{args.split}.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved: {summary_path}")

    iou_keys = [f"AP@{int(t*100)}" for t in IOU_THRESHOLDS] + ["mAP@50:95"]
    csv_path = out_dir / f"map_multi_threshold_{args.split}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model"] + iou_keys)
        for model_name, results in all_results.items():
            writer.writerow([model_name] + [results.get(k, "") for k in iou_keys])
    print(f"Saved: {csv_path}")

    print("\n" + "="*70)
    print(f"{'Model':<30} " + " ".join(f"AP@{int(t*100):2d}" for t in HIGHLIGHT_IOUS))
    print("-"*70)
    for model_name, results in all_results.items():
        vals = " ".join(f"{results.get(f'AP@{int(t*100)}', 0):6.2f}" for t in HIGHLIGHT_IOUS)
        print(f"{model_name:<30} {vals}")
    print("="*70)

    plot_comparison(all_results, out_dir / f"map_bar_chart_{args.split}.png")


if __name__ == "__main__":
    main()
