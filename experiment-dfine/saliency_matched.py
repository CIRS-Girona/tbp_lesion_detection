#!/usr/bin/env python3
"""
saliency_matched.py
===================
Generate EigenCAM activation maps for BOTH YOLOv12s AND D-FINE-S
on the SAME set of images — fixing Hayat's comment that the two
models were visualised on different images.

Strategy
--------
1. Scan the test set for images where BOTH models detect at least one
   lesion above --min-conf.
2. Pick --n-images such shared images (or use --image-ids to specify
   exact filenames / COCO image IDs directly).
3. Run EigenCAM for YOLO and for D-FINE on every selected image.
4. Save side-by-side panels to --output-dir (one subfolder per model,
   plus a combined 2×2 comparison panel).

Usage
-----
# Auto-select 6 images where both models detect something:
python scripts/saliency_matched.py \
    --yolo-weights  ~/path/to/yolov12s_best.pt \
    --dfine-config  ~/path/to/dfine_s_fullaug_sweep_100ep.yml \
    --dfine-weights ~/path/to/best_stg2.pth \
    --dfine-repo    ~/path/to/D-FINE \
    --dataset-dir   ~/path/to/dataset_coco \
    --split test \
    --n-images 6 \
    --min-conf 0.25 \
    --device cuda:0 \
    --output-dir results/saliency_matched

# Pin specific images (by filename stem or COCO image id):
python scripts/saliency_matched.py ... \
    --image-ids test_000017_image_0070 test_000001_image_0009

Requirements
------------
pip install grad-cam ultralytics Pillow matplotlib
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Matched EigenCAM for YOLO + D-FINE")
    p.add_argument("--yolo-weights",  type=Path, required=True,
                   help="Path to YOLOv12s best.pt")
    p.add_argument("--dfine-config",  type=Path, required=True,
                   help="Path to D-FINE YAML config")
    p.add_argument("--dfine-weights", type=Path, required=True,
                   help="Path to D-FINE best_stg2.pth")
    p.add_argument("--dfine-repo",    type=Path, required=True,
                   help="Path to D-FINE git repository (for src/core)")
    p.add_argument("--dataset-dir",   type=Path, required=True,
                   help="COCO dataset root (must have annotations/ and images/)")
    p.add_argument("--split",         choices=["val", "test"], default="test")
    p.add_argument("--n-images",      type=int, default=6,
                   help="Number of matched images to generate")
    p.add_argument("--min-conf",      type=float, default=0.25,
                   help="Min confidence to count as detection")
    p.add_argument("--img-size",      type=int, default=1024)
    p.add_argument("--device",        type=str, default="cuda:0",
                   help="Device for D-FINE inference (YOLO EigenCAM always on CPU)")
    p.add_argument("--output-dir",    type=Path,
                   default=Path("results/saliency_matched"))
    p.add_argument("--image-ids",     nargs="+", default=None,
                   help="Optional list of image filename stems to force-select "
                        "(overrides auto-selection). E.g. test_000017_image_0070")
    return p.parse_args()


def load_coco_images(dataset_dir: Path, split: str):
    ann_file = dataset_dir / "annotations" / f"instances_{split}.json"
    data = json.loads(ann_file.read_text())
    return data["images"], dataset_dir / "images" / split


def yolo_has_detections(yolo_model, img_path: Path, img_size: int,
                         min_conf: float) -> bool:
    import numpy as np
    from PIL import Image
    img = np.array(Image.open(img_path).convert("RGB"))
    results = yolo_model.predict(img, imgsz=img_size, device="cpu",
                                  conf=min_conf, verbose=False)
    return bool(results and len(results[0].boxes) > 0)


def dfine_has_detections(model_deploy, postproc, img_path: Path,
                          img_size: int, min_conf: float, device) -> bool:
    import torch
    import torchvision.transforms as T
    from PIL import Image as PILImage
    img = PILImage.open(img_path).convert("RGB")
    w, h = img.size
    transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    tensor = transform(img).unsqueeze(0).to(device)
    orig_size = torch.tensor([[w, h]], device=device)
    with torch.no_grad():
        _, _, scores = postproc(model_deploy(tensor), orig_size)
    return bool((scores[0].cpu().numpy() >= min_conf).any())


def run_yolo_eigencam(yolo_model, img_path: Path, img_size: int,
                       min_conf: float, out_path: Path) -> bool:
    """Run EigenCAM for YOLOv12 on a single image. Returns True if saved."""
    import torch
    from PIL import Image
    import torchvision.transforms as T
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    try:
        from pytorch_grad_cam import EigenCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("[ERROR] pip install grad-cam")
        return False

    model = yolo_model.model
    model.eval()

    # Target: last CV layer in detection head
    target_layers = []
    for name, module in model.named_modules():
        if "detect" in name.lower() or name.split(".")[-1] in ["cv2", "cv3"]:
            if hasattr(module, "weight"):
                target_layers.append(module)
    if not target_layers:
        conv_layers = [(n, m) for n, m in model.named_modules()
                       if isinstance(m, torch.nn.Conv2d)]
        if conv_layers:
            target_layers = [conv_layers[-3][1]]
    if not target_layers:
        print(f"[ERROR] Could not find YOLO target layer")
        return False

    class YOLOWrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.model = m
        def forward(self, x):
            out = self.model(x)
            if isinstance(out, (list, tuple)):
                for item in out:
                    if isinstance(item, torch.Tensor): return item
                return out[0]
            return out

    wrapped = YOLOWrapper(model)
    cam = EigenCAM(model=wrapped, target_layers=[target_layers[-1]])

    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img.resize((img_size, img_size))) / 255.0
    transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    input_tensor = transform(img).unsqueeze(0)

    results = yolo_model.predict(np.array(img), imgsz=img_size, device="cpu",
                                  conf=min_conf, verbose=False)
    if not results or len(results[0].boxes) == 0:
        return False

    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()

    try:
        grayscale_cam = cam(input_tensor=input_tensor)[0]
    except Exception as e:
        print(f"  [warn] YOLO EigenCAM failed: {e}")
        return False

    cam_img = show_cam_on_image(img_np.astype(np.float32), grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_np); axes[0].set_title("Input + YOLOv12s Detections"); axes[0].axis("off")
    for box, conf in zip(boxes, confs):
        scale_x = img_size / img.width;  scale_y = img_size / img.height
        x1, y1, x2, y2 = [int(v) for v in box]
        rect = patches.Rectangle((x1/scale_x, y1/scale_y),
                                  (x2-x1)/scale_x, (y2-y1)/scale_y,
                                  linewidth=2, edgecolor="lime", facecolor="none")
        axes[0].add_patch(rect)
        axes[0].text(x1/scale_x, y1/scale_y - 3, f"{conf:.2f}",
                     color="lime", fontsize=8, fontweight="bold")

    axes[1].imshow(cam_img)
    axes[1].set_title("YOLOv12s EigenCAM — Detection Head\n"
                       "(red = high activation → which pixels trigger detection)")
    axes[1].axis("off")

    plt.suptitle(f"YOLOv12s EigenCAM — {img_path.stem}", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def run_dfine_eigencam(model_cam, model_deploy, postproc, img_path: Path,
                        img_size: int, min_conf: float, device,
                        out_path: Path) -> bool:
    """Run EigenCAM for D-FINE on a single image. Returns True if saved."""
    import torch
    import torchvision.transforms as T
    from PIL import Image as PILImage
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    try:
        from pytorch_grad_cam import EigenCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("[ERROR] pip install grad-cam")
        return False

    img = PILImage.open(img_path).convert("RGB")
    w, h = img.size
    img_np = np.array(img.resize((img_size, img_size))) / 255.0
    transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    tensor_cpu = transform(img).unsqueeze(0)
    tensor_gpu = tensor_cpu.to(device)
    orig_size  = torch.tensor([[w, h]], device=device)

    with torch.no_grad():
        labels, boxes_t, scores_t = postproc(model_deploy(tensor_gpu), orig_size)
    scores_np = scores_t[0].cpu().numpy()
    boxes_np  = boxes_t[0].cpu().numpy()
    high_conf = scores_np >= min_conf
    if not high_conf.any():
        return False

    try:
        grayscale_cam = model_cam(input_tensor=tensor_cpu)[0]
    except Exception as e:
        print(f"  [warn] D-FINE EigenCAM failed: {e}")
        return False

    cam_img = show_cam_on_image(img_np.astype(np.float32), grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_np); axes[0].set_title("Input + D-FINE-S Detections"); axes[0].axis("off")
    for box, conf in zip(boxes_np[high_conf], scores_np[high_conf]):
        x1 = int(box[0] * img_size / w);  y1 = int(box[1] * img_size / h)
        x2 = int(box[2] * img_size / w);  y2 = int(box[3] * img_size / h)
        rect = mpatches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=2, edgecolor="lime", facecolor="none")
        axes[0].add_patch(rect)
        axes[0].text(x1, y1-3, f"{conf:.2f}", color="lime", fontsize=8, fontweight="bold")

    axes[1].imshow(cam_img)
    axes[1].set_title("D-FINE-S EigenCAM — HGNetv2 Backbone\n"
                       "(red = high activation → Clever Hans check)")
    axes[1].axis("off")

    plt.suptitle(f"D-FINE-S EigenCAM — {img_path.stem}", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def make_combined_panel(yolo_path: Path, dfine_path: Path, out_path: Path,
                         stem: str) -> None:
    """Combine YOLO and D-FINE side-by-side (2 rows × 2 cols) into one figure."""
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    yolo_img  = mpimg.imread(str(yolo_path))
    dfine_img = mpimg.imread(str(dfine_path))

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    axes[0].imshow(yolo_img);  axes[0].axis("off"); axes[0].set_title("YOLOv12s",  fontsize=12)
    axes[1].imshow(dfine_img); axes[1].axis("off"); axes[1].set_title("D-FINE-S",  fontsize=12)

    plt.suptitle(f"EigenCAM Comparison — {stem}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [combined] {out_path.name}")


def main() -> None:
    args = parse_args()

    args.yolo_weights  = args.yolo_weights.expanduser().resolve()
    args.dfine_config  = args.dfine_config.expanduser().resolve()
    args.dfine_weights = args.dfine_weights.expanduser().resolve()
    args.dfine_repo    = args.dfine_repo.expanduser().resolve()
    args.dataset_dir   = args.dataset_dir.expanduser().resolve()
    out_dir            = args.output_dir.expanduser().resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    yolo_dir  = out_dir / "eigencam_yolo"
    dfine_dir = out_dir / "eigencam_dfine"
    comb_dir  = out_dir / "combined"
    for d in [yolo_dir, dfine_dir, comb_dir]:
        d.mkdir(parents=True, exist_ok=True)

    images, img_folder = load_coco_images(args.dataset_dir, args.split)

    if args.image_ids:
        id_set = set(args.image_ids)
        images = [img for img in images
                  if Path(img["file_name"]).stem in id_set
                  or str(img["id"]) in id_set]
        print(f"[info] Forced image selection: {len(images)} images matched --image-ids")
        if not images:
            print("[ERROR] No images matched the provided --image-ids. Check stems.")
            sys.exit(1)

    print("\n=== Loading YOLOv12s ===")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] pip install ultralytics")
        sys.exit(1)
    yolo_model = YOLO(str(args.yolo_weights))
    print(f"  Loaded: {args.yolo_weights.name}")

    print("\n=== Loading D-FINE-S ===")
    sys.path.insert(0, str(args.dfine_repo))
    import torch

    try:
        from src.core import YAMLConfig  # type: ignore
    except ImportError:
        print(f"[ERROR] Cannot import D-FINE src — check --dfine-repo: {args.dfine_repo}")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Model for EigenCAM (CPU, needs .eval(), no deploy() wrapper)
    cfg_cam = YAMLConfig(str(args.dfine_config), resume=str(args.dfine_weights))
    if "HGNetv2" in cfg_cam.yaml_cfg:
        cfg_cam.yaml_cfg["HGNetv2"]["pretrained"] = False
    ckpt = torch.load(str(args.dfine_weights), map_location="cpu")
    state = ckpt.get("ema", {}).get("module", ckpt.get("model", ckpt))
    cfg_cam.model.load_state_dict(state, strict=False)
    dfine_model_cpu = cfg_cam.model.eval()

    # Find backbone target layer for EigenCAM
    backbone_convs = [(n, m) for n, m in dfine_model_cpu.named_modules()
                      if isinstance(m, torch.nn.Conv2d) and "backbone" in n]
    if not backbone_convs:
        print("[ERROR] No backbone Conv2d found in D-FINE")
        sys.exit(1)
    target_layer = [backbone_convs[-1][1]]
    print(f"  EigenCAM target: {backbone_convs[-1][0]}")

    class DFineWrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.model = m
        def forward(self, x):
            out = self.model(x)
            if isinstance(out, dict):
                return out.get("pred_logits", torch.zeros(1, 1, 1))
            return out

    from pytorch_grad_cam import EigenCAM
    dfine_cam = EigenCAM(model=DFineWrapper(dfine_model_cpu), target_layers=target_layer)

    # Deploy model for D-FINE predictions (on device)
    cfg_deploy = YAMLConfig(str(args.dfine_config), resume=str(args.dfine_weights))
    if "HGNetv2" in cfg_deploy.yaml_cfg:
        cfg_deploy.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg_deploy.model.load_state_dict(state, strict=False)
    dfine_deploy  = cfg_deploy.model.deploy().to(device).eval()
    dfine_postpro = cfg_deploy.postprocessor.deploy().to(device)
    print("  D-FINE loaded successfully")

    print(f"\n=== Selecting up to {args.n_images} images where BOTH models detect ===")
    selected = []

    for img_info in images:
        if len(selected) >= args.n_images:
            break
        img_path = img_folder / img_info["file_name"]
        if not img_path.exists():
            continue

        yolo_ok  = yolo_has_detections(yolo_model, img_path, args.img_size, args.min_conf)
        dfine_ok = dfine_has_detections(dfine_deploy, dfine_postpro, img_path,
                                         args.img_size, args.min_conf, device)

        status = f"YOLO={'✓' if yolo_ok else '✗'}  D-FINE={'✓' if dfine_ok else '✗'}"
        if yolo_ok and dfine_ok:
            selected.append(img_info)
            print(f"  [SELECT] {img_info['file_name']}  {status}")
        else:
            print(f"  [skip]   {img_info['file_name']}  {status}")

    if not selected:
        print("[ERROR] No images found where both models detect. "
              "Try lowering --min-conf or expanding the image list.")
        sys.exit(1)

    print(f"\nSelected {len(selected)} matched images.")

    print("\n=== Generating EigenCAM maps ===")
    generated = 0

    for i, img_info in enumerate(selected, 1):
        img_path = img_folder / img_info["file_name"]
        stem     = img_path.stem

        yolo_out  = yolo_dir  / f"yolo_eigencam_{stem}.png"
        dfine_out = dfine_dir / f"dfine_eigencam_{stem}.png"
        comb_out  = comb_dir  / f"combined_{stem}.png"

        print(f"\n[{i}/{len(selected)}] {stem}")

        yolo_ok = run_yolo_eigencam(
            yolo_model, img_path, args.img_size, args.min_conf, yolo_out
        )
        dfine_ok = run_dfine_eigencam(
            dfine_cam, dfine_deploy, dfine_postpro,
            img_path, args.img_size, args.min_conf, device, dfine_out
        )

        if yolo_ok and dfine_ok:
            make_combined_panel(yolo_out, dfine_out, comb_out, stem)
            generated += 1
            print(f"  ✓ Saved YOLO: {yolo_out.name}")
            print(f"  ✓ Saved D-FINE: {dfine_out.name}")
        else:
            print(f"  [warn] Skipped combined panel (YOLO={yolo_ok}, D-FINE={dfine_ok})")

    print(f"\n{'='*60}")
    print(f"Done! Generated {generated} matched EigenCAM pairs.")
    print(f"Output directory: {out_dir}")
    print(f"  eigencam_yolo/  — YOLOv12s panels")
    print(f"  eigencam_dfine/ — D-FINE-S panels")
    print(f"  combined/       — side-by-side comparison panels")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
