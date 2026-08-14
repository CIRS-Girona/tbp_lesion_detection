#!/usr/bin/env python
"""
eval_f1_pr.py
-------------
Full detection evaluation for the official RT-DETRv2 repo (works for BOTH the
baseline `model: RTDETR` config and the `model: RTDETRSCALE` config).

It:
  1. runs the model on the eval split and dumps predictions to COCO JSON
  2. recomputes COCO mAP50 / mAP50-95 (sanity check vs --test-only)
  3. sweeps the confidence threshold and reports the BEST F1 (+ P, R) at IoU=0.5
     -> this is the team's F1 definition: one operating point on the PR curve
  4. saves a PR curve plot and an F1-vs-confidence plot
  5. writes the confidence sweep to CSV

The split is decided by the config's val_dataloader, so pass the *_test.yml to
evaluate on TEST. predictions JSON is reused by tide_analysis.py.

Run FROM THE REPO ROOT (rtdetrv2_pytorch/):
  CUDA_VISIBLE_DEVICES=1 python eval_f1_pr.py \
      -c configs/rtdetrv2/rtdetrv2_r18vd_itobos_test.yml \
      -r /path/to/rtdetrv2_output/rtdetrv2_r18vd_itobos_1024/best.pth \
      --tag baseline_test

Author: Praveen Kumar Murali | June 2026
"""
import os
import sys
import json
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from src.core import YAMLConfig


def gather_predictions(cfg, device):
    """Run inference over cfg.val_dataloader, return COCO-format detections."""
    model = cfg.model
    model.eval().to(device)
    post = cfg.postprocessor
    post.eval()
    loader = cfg.val_dataloader

    coco_pred = []
    n_img = 0
    with torch.no_grad():
        for samples, targets in loader:
            samples = samples.to(device)
            orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
            outputs = model(samples)
            results = post(outputs, orig_sizes)          # list of {labels,boxes,scores}
            for t, r in zip(targets, results):
                image_id = int(t["image_id"].item())
                boxes = r["boxes"].cpu().numpy()          # xyxy in original pixels
                scores = r["scores"].cpu().numpy()
                labels = r["labels"].cpu().numpy()
                for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels):
                    coco_pred.append({
                        "image_id": image_id,
                        "category_id": int(l),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(s),
                    })
                n_img += 1
    print(f"[pred] {n_img} images, {len(coco_pred)} raw detections")
    return coco_pred


def coco_map(gt_json, pred_json):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(pred_json)
    e = COCOeval(coco_gt, coco_dt, "bbox")
    e.evaluate(); e.accumulate(); e.summarize()
    return {"mAP50_95": float(e.stats[0]), "mAP50": float(e.stats[1]), "mAP75": float(e.stats[2])}


def iou_xywh(a, boxes):
    """IoU of one [x,y,w,h] box vs Nx4 [x,y,w,h] boxes."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1 = boxes[:, 0], boxes[:, 1]
    bx2, by2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    ix1 = np.maximum(ax1, bx1); iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2); iy2 = np.minimum(ay2, by2)
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    union = a[2] * a[3] + boxes[:, 2] * boxes[:, 3] - inter
    return np.where(union > 0, inter / union, 0.0)


def f1_sweep(gt_json, preds, confs, iou_thr=0.5):
    """Greedy 1-to-1 matching per image (single class) across confidence thresholds."""
    from pycocotools.coco import COCO
    coco = COCO(gt_json)
    img_ids = coco.getImgIds()

    gt_by_img = {}
    for iid in img_ids:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=iid))
        gt_by_img[iid] = (np.array([an["bbox"] for an in anns], dtype=float)
                          if anns else np.zeros((0, 4)))
    pred_by_img = {iid: [] for iid in img_ids}
    for p in preds:
        if p["image_id"] in pred_by_img:
            pred_by_img[p["image_id"]].append((p["score"], p["bbox"]))
    for iid in pred_by_img:
        pred_by_img[iid].sort(key=lambda z: -z[0])          # high score first

    rows = []
    for c in confs:
        TP = FP = FN = 0
        for iid in img_ids:
            gts = gt_by_img[iid]
            used = np.zeros(len(gts), dtype=bool)
            for s, b in pred_by_img[iid]:
                if s < c:
                    break                                    # sorted -> rest are lower
                if len(gts) == 0:
                    FP += 1; continue
                ious = iou_xywh(np.asarray(b), gts)
                ious[used] = -1
                j = int(np.argmax(ious))
                if ious[j] >= iou_thr:
                    TP += 1; used[j] = True
                else:
                    FP += 1
            FN += int((~used).sum())
        P = TP / (TP + FP) if TP + FP else 0.0
        R = TP / (TP + FN) if TP + FN else 0.0
        F1 = 2 * P * R / (P + R) if P + R else 0.0
        rows.append((c, P, R, F1, TP, FP, FN))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-r", "--resume", required=True)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--outdir", default="eval_out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cfg = YAMLConfig(args.config)
    ckpt = torch.load(args.resume, map_location="cpu")
    if "ema" in ckpt and ckpt["ema"] is not None:
        state = ckpt["ema"]["module"]; print("[ckpt] using EMA weights")
    else:
        state = ckpt["model"]; print("[ckpt] using model weights")
    cfg.model.load_state_dict(state)

    gt_json = cfg.yaml_cfg["val_dataloader"]["dataset"]["ann_file"]
    print(f"[gt]  {gt_json}")

    preds = gather_predictions(cfg, args.device)
    pred_path = os.path.join(args.outdir, f"pred_{args.tag}.json")
    json.dump(preds, open(pred_path, "w"))
    print(f"[pred] saved -> {pred_path}")

    print("\n=== COCO mAP (sanity, should match --test-only) ===")
    m = coco_map(gt_json, pred_path)

    confs = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2,
             0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]
    rows = f1_sweep(gt_json, preds, confs, args.iou)
    best = max(rows, key=lambda z: z[3])

    print(f"\n=== F1 sweep (IoU={args.iou:.2f}) ===")
    print(f"{'conf':>6} {'P':>7} {'R':>7} {'F1':>7} {'TP':>6} {'FP':>6} {'FN':>6}")
    for (c, P, R, F1, TP, FP, FN) in rows:
        mark = "  <-- best" if (c, P, R, F1) == best[:4] else ""
        print(f"{c:6.3f} {P:7.3f} {R:7.3f} {F1:7.3f} {TP:6d} {FP:6d} {FN:6d}{mark}")

    csv_path = os.path.join(args.outdir, f"f1_sweep_{args.tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["conf", "P", "R", "F1", "TP", "FP", "FN"]); w.writerows(rows)

    print(f"\n========== SUMMARY [{args.tag}] ==========")
    print(f"mAP50      = {m['mAP50']:.3f}")
    print(f"mAP50-95   = {m['mAP50_95']:.3f}")
    print(f"mAP75      = {m['mAP75']:.3f}")
    print(f"best F1    = {best[3]:.3f}  (conf={best[0]:.3f}, P={best[1]:.3f}, R={best[2]:.3f}, IoU={args.iou})")
    print(f"[csv] {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cs = [r[0] for r in rows]; Ps = [r[1] for r in rows]
        Rs = [r[2] for r in rows]; Fs = [r[3] for r in rows]
        plt.figure(); plt.plot(Rs, Ps, marker="o")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.grid(True)
        plt.title(f"PR curve ({args.tag})")
        plt.savefig(os.path.join(args.outdir, f"pr_{args.tag}.png"), dpi=150, bbox_inches="tight")
        plt.figure(); plt.plot(cs, Fs, marker="o")
        plt.xlabel("confidence"); plt.ylabel("F1"); plt.grid(True)
        plt.title(f"F1 vs confidence ({args.tag})")
        plt.savefig(os.path.join(args.outdir, f"f1conf_{args.tag}.png"), dpi=150, bbox_inches="tight")
        print(f"[plot] pr_{args.tag}.png, f1conf_{args.tag}.png in {args.outdir}/")
    except Exception as e:
        print(f"[plot] skipped ({e})")


if __name__ == "__main__":
    main()
