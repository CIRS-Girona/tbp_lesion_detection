#!/usr/bin/env python3
"""
Saliency / Attention Map Analysis
===================================
Hayat's email justification:
  "A detector can emit a correct box while relying on the background context
   rather than the object — the 'Clever Hans' effect, which matters a lot for
   lesion detection. Such pixel-based attribution methods answer 'which pixels
   resulted in this box,' not 'is the box right'."

This script implements THREE approaches (pick the one that works):

1. EigenCAM on YOLO detection head — fast, no backward pass needed
   Ref: Muhammad et al., "Eigen-CAM: Class Activation Map using Principal
   Components of Convolutional Layers", IJCNN 2020
   Library: pytorch-grad-cam (pip install grad-cam)

2. D-FINE Decoder Cross-Attention Maps — extract attention from decoder
   cross-attention layers (no external library needed)

3. ODAM (Object Detection Attention Map) — gradient-based per-detection
   Ref: Zhao et al., "ODAM: Visualization and Explanation of Object Detection
   Models", ICLR 2023 workshop
   Not a full implementation; uses GradCAM with detection-specific weighting.

Usage:
  # EigenCAM for YOLOv12:
  python scripts/saliency_analysis.py \
    --mode eigen_cam \
    --yolo-weights /path/to/yolov12s.pt \
    --dataset-dir /path/to/dataset_coco \
    --split test \
    --n-images 10 \
    --output-dir results/saliency/

  # D-FINE cross-attention:
  python scripts/saliency_analysis.py \
    --mode dfine_attn \
    --dfine-config results/configs/dfine_s_fullaug_sweep_100ep.yml \
    --dfine-weights results/training/full/dfine_s_fullaug_sweep_100ep/best_stg2.pth \
    --dataset-dir /path/to/dataset_coco \
    --split test \
    --n-images 10 \
    --output-dir results/saliency/

  # Both:
  python scripts/saliency_analysis.py \
    --mode both ...
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

DFINE_REPO  = Path(os.environ.get("DFINE_REPO", "~/D-FINE")).expanduser()
DATASET_DIR = Path(os.environ.get("DFINE_DATASET_DIR", "~/dfine_results/dataset_coco")).expanduser()
RESULTS_DIR = Path(os.environ.get("DFINE_RESULTS_DIR", "results")).expanduser()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode",          choices=["eigen_cam", "dfine_attn", "both"],
                   default="both")
    p.add_argument("--dfine-repo",    type=Path, default=DFINE_REPO)
    p.add_argument("--dataset-dir",   type=Path, default=DATASET_DIR)
    p.add_argument("--output-dir",    type=Path,
                   default=RESULTS_DIR / "saliency")
    p.add_argument("--dfine-config",  type=Path, default=None)
    p.add_argument("--dfine-weights", type=Path, default=None)
    p.add_argument("--yolo-weights",  type=Path, default=None)
    p.add_argument("--split",         choices=["val","test"], default="test")
    p.add_argument("--n-images",      type=int, default=10)
    p.add_argument("--img-size",      type=int, default=1024)
    p.add_argument("--device",        type=str, default="cuda:0")
    p.add_argument("--min-conf",      type=float, default=0.25)
    return p.parse_args()


def run_eigen_cam_yolo(args: argparse.Namespace, images: list,
                        img_folder: Path) -> None:
    """
    EigenCAM on YOLO detection head.
    Requires: pip install grad-cam ultralytics Pillow
    """
    import torch
    from PIL import Image
    import torchvision.transforms as T

    try:
        from ultralytics import YOLO  # type: ignore
        from pytorch_grad_cam import EigenCAM  # type: ignore
        from pytorch_grad_cam.utils.image import show_factorization_on_image  # type: ignore
    except ImportError:
        print("[ERROR] Install: pip install grad-cam ultralytics")
        return

    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("[ERROR] Install: pip install matplotlib")
        return

    out_dir = args.output_dir / "eigencam_yolo"
    out_dir.mkdir(parents=True, exist_ok=True)

    yolo_model = YOLO(str(args.yolo_weights.expanduser().resolve()))
    model = yolo_model.model
    model.eval()

    # Target layer: last layer of the detection head
    # For YOLOv12, this is typically model.model[-1] (Detect head)
    target_layers = []
    for name, module in model.named_modules():
        if "detect" in name.lower() or name.split(".")[-1] in ["cv2", "cv3"]:
            if hasattr(module, "weight"):
                target_layers.append(module)

    if not target_layers:
        # Fallback: use the second-to-last Conv layer
        conv_layers = [(n, m) for n, m in model.named_modules()
                       if isinstance(m, torch.nn.Conv2d)]
        if conv_layers:
            target_layers = [conv_layers[-3][1]]

    if not target_layers:
        print("[ERROR] Could not find target layer for EigenCAM")
        return

    print(f"EigenCAM target: {target_layers[-1].__class__.__name__}")
    class YOLOModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            out = self.model(x)
            if isinstance(out, (list, tuple)):
                for item in out:
                    if isinstance(item, torch.Tensor):
                        return item
                return out[0]
            return out

    wrapped_model = YOLOModelWrapper(model)
    cam = EigenCAM(model=wrapped_model, target_layers=[target_layers[-1]])

    transform = T.Compose([T.Resize((args.img_size, args.img_size)), T.ToTensor()])

    processed = 0
    for img_info in images:
        if processed >= args.n_images:
            break
        img_path = img_folder / img_info["file_name"]
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img.resize((args.img_size, args.img_size))) / 255.0
        input_tensor = transform(img).unsqueeze(0)

        results = yolo_model.predict(
            np.array(img), imgsz=args.img_size, device="cpu",
            conf=args.min_conf, verbose=False
        )
        if not results or len(results[0].boxes) == 0:
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        try:
            grayscale_cam = cam(input_tensor=input_tensor)
            grayscale_cam = grayscale_cam[0]  # [H, W]
        except Exception as e:
            print(f"EigenCAM failed for {img_info['file_name']}: {e}")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img_np)
        axes[0].set_title("Original + Detections")
        axes[0].axis("off")
        for box, conf in zip(boxes, confs):
            scale_x = args.img_size / img.width
            scale_y = args.img_size / img.height
            x1, y1, x2, y2 = [int(v) for v in box]
            rect = patches.Rectangle(
                (x1/scale_x, y1/scale_y),
                (x2-x1)/scale_x, (y2-y1)/scale_y,
                linewidth=2, edgecolor="lime", facecolor="none"
            )
            axes[0].add_patch(rect)
            axes[0].text(x1/scale_x, y1/scale_y - 3, f"{conf:.2f}",
                         color="lime", fontsize=8)

        axes[1].imshow(img_np)
        axes[1].imshow(grayscale_cam, alpha=0.5, cmap="jet")
        axes[1].set_title("EigenCAM — Detection Head Saliency\n"
                           "(red=high activation → which pixels trigger detection)")
        axes[1].axis("off")

        fname = Path(img_info["file_name"]).stem
        plt.suptitle(f"YOLOv12s EigenCAM: {fname}", fontsize=10)
        plt.tight_layout()
        plt.savefig(out_dir / f"eigencam_{fname}.png", dpi=150, bbox_inches="tight")
        plt.close()
        processed += 1
        print(f"  [{processed}/{args.n_images}] Saved: eigencam_{fname}.png")

    print(f"\nEigenCAM complete. Saved {processed} images to: {out_dir}")


def run_dfine_cross_attention(args: argparse.Namespace, images: list,
                               img_folder: Path) -> None:
    """
    Extract decoder cross-attention maps from D-FINE.

    D-FINE calls cross_attn(q, k, v, need_weights=False) for inference speed.
    We monkey-patch every nn.MultiheadAttention in the decoder to force
    need_weights=True so attention weights are actually computed and captured.
    """
    import sys
    sys.path.insert(0, str(args.dfine_repo.expanduser().resolve()))
    import torch
    import torchvision.transforms as T
    from PIL import Image as PILImage
    from src.core import YAMLConfig  # type: ignore

    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[ERROR] pip install matplotlib")
        return

    out_dir = args.output_dir / "dfine_cross_attention"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = args.dfine_config.expanduser().resolve()
    wt_path  = args.dfine_weights.expanduser().resolve()
    cfg = YAMLConfig(str(cfg_path), resume=str(wt_path))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    ckpt  = torch.load(str(wt_path), map_location="cpu")
    state = ckpt.get("ema", {}).get("module", ckpt.get("model", ckpt))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load ONE model for attention capture (non-deploy, so ops are not fused)
    model_attn = cfg.model
    model_attn.load_state_dict(state, strict=False)
    model_attn = model_attn.to(device).eval()

    # Load a SEPARATE deploy model for final predictions
    cfg2 = YAMLConfig(str(cfg_path), resume=str(wt_path))
    if "HGNetv2" in cfg2.yaml_cfg:
        cfg2.yaml_cfg["HGNetv2"]["pretrained"] = False
    model_deploy = cfg2.model
    model_deploy.load_state_dict(state, strict=False)
    postproc = cfg2.postprocessor.deploy().to(device)
    model_deploy = model_deploy.deploy().to(device).eval()

    # Monkey-patch decoder MHA to force need_weights=True.
    # D-FINE's TransformerDecoderLayer calls cross_attn with need_weights=False,
    # which means attn weights are never computed. We override this.
    attn_store: dict = {}          # layer_name → [B, heads, N_q, N_k]
    module_originals: dict = {}    # id(module) → (module, original_forward)

    def _make_capturing_fwd(layer_name: str, orig_fwd):
        def capturing_fwd(query, key=None, value=None, *args, **kwargs):
            kwargs["need_weights"] = True
            kwargs.pop("average_attn_weights", None)  # keep per-head weights
            result = orig_fwd(query, key, value, *args, **kwargs)
            out, weights = result if isinstance(result, tuple) else (result, None)
            if weights is not None:
                attn_store[layer_name] = weights.detach().cpu()
            return out, weights
        return capturing_fwd

    n_patched = 0
    for name, module in model_attn.named_modules():
        # Must have BOTH "decoder" AND "cross_attn" in name.
        # self_attn is also nn.MultiheadAttention in the decoder but we don't
        # want it — its N_k equals N_q (queries), not H*W of the encoder map.
        # D-FINE's real cross-attention is MSDeformAttn (not nn.MultiheadAttention)
        # so n_patched will likely be 0, triggering the EigenCAM fallback below.
        if (isinstance(module, torch.nn.MultiheadAttention) and
                "decoder" in name.lower() and
                "cross_attn" in name.lower()):
            module_originals[id(module)] = (module, module.forward)
            module.forward = _make_capturing_fwd(name, module.forward)
            n_patched += 1
            print(f"  [patch] {name}")

    if n_patched == 0:
        # D-FINE uses MSDeformAttn for cross-attention (not nn.MultiheadAttention).
        # Deformable attention doesn't expose standard [N_q × H*W] weights.
        # Fall back to EigenCAM on the backbone — fully valid for Clever Hans check.
        print("  [info] D-FINE uses deformable cross-attention (MSDeformAttn).")
        print("  [info] Standard attention weights unavailable — using EigenCAM fallback.")
        _dfine_eigencam_fallback(args, images, img_folder, out_dir)
        return

    print(f"  Patched {n_patched} decoder cross-attention module(s).")

    transform = T.Compose([T.Resize((args.img_size, args.img_size)), T.ToTensor()])

    processed = 0
    for img_info in images:
        if processed >= args.n_images:
            break

        attn_store.clear()
        img_path = img_folder / img_info["file_name"]
        if not img_path.exists():
            continue
        img = PILImage.open(img_path).convert("RGB")
        w, h = img.size
        img_np = np.array(img.resize((args.img_size, args.img_size))) / 255.0

        tensor    = transform(img).unsqueeze(0).to(device)
        orig_size = torch.tensor([[w, h]], device=device)

        # Forward through attention-patched model to fill attn_store
        with torch.no_grad():
            model_attn(tensor)

        # Forward through deploy model to get clean predictions
        with torch.no_grad():
            labels, boxes, scores = postproc(model_deploy(tensor), orig_size)

        scores_np = scores[0].cpu().numpy()
        boxes_np  = boxes[0].cpu().numpy()
        high_conf = scores_np >= args.min_conf

        if not high_conf.any():
            print(f"  [skip] No detections >= {args.min_conf:.2f} in "
                  f"{img_info['file_name']}")
            continue

        if not attn_store:
            print(f"  [warn] Patches active but no weights captured — "
                  f"D-FINE may use a custom fused kernel.")
            print("  [info] Falling back to backbone GradCAM...")
            _dfine_gradcam_fallback(args, images, img_folder, out_dir)
            break

        # Use the LAST decoder layer (most refined attention)
        last_key = sorted(attn_store.keys())[-1]
        raw = attn_store[last_key]
        # nn.MultiheadAttention returns:
        #   average_attn_weights=True (default): [B, N_q, N_k]
        #   average_attn_weights=False:          [B, heads, N_q, N_k]
        if raw.dim() == 4:
            attn = raw[0].mean(0)   # [heads, N_q, N_k] → mean → [N_q, N_k]
        elif raw.dim() == 3:
            attn = raw[0]           # [N_q, N_k]  (already averaged over heads)
        else:
            print(f"  [warn] Unexpected attn shape {raw.shape}, skipping image.")
            continue

        N_q, N_k = attn.shape
        feat_h = feat_w = int(np.sqrt(N_k))
        if feat_h * feat_w != N_k:
            for fh in range(int(np.sqrt(N_k)), 0, -1):
                if N_k % fh == 0:
                    feat_h, feat_w = fh, N_k // fh
                    break
        assert attn.shape == (N_q, N_k), f"Shape mismatch: {attn.shape}"

        n_dets = min(3, int(high_conf.sum()))
        fig, axes = plt.subplots(1, n_dets + 1,
                                  figsize=(5 * (n_dets + 1), 5))
        if n_dets + 1 == 1:
            axes = [axes]

        axes[0].imshow(img_np)
        axes[0].set_title("Input + Detections", fontsize=9)
        axes[0].axis("off")
        for box, conf in zip(boxes_np[high_conf], scores_np[high_conf]):
            x1 = int(box[0] * args.img_size / w)
            y1 = int(box[1] * args.img_size / h)
            x2 = int(box[2] * args.img_size / w)
            y2 = int(box[3] * args.img_size / h)
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="lime", facecolor="none"
            )
            axes[0].add_patch(rect)
            axes[0].text(x1, y1 - 3, f"{conf:.2f}",
                         color="lime", fontsize=8, fontweight="bold")

        # Match each detection box to the decoder query whose attention peak
        # is spatially closest to the box centre.
        high_idx = np.where(high_conf)[0][:n_dets]
        for col, det_idx in enumerate(high_idx, 1):
            box  = boxes_np[det_idx]
            # Box centre in feature-map coordinates
            cx = (box[0] + box[2]) / 2 / w * feat_w
            cy = (box[1] + box[3]) / 2 / h * feat_h

            best_q, best_dist = 0, float("inf")
            for q in range(attn.shape[0]):
                q_map = attn[q].numpy().reshape(feat_h, feat_w)
                peak  = np.unravel_index(q_map.argmax(), q_map.shape)
                dist  = (peak[1] - cx) ** 2 + (peak[0] - cy) ** 2
                if dist < best_dist:
                    best_dist, best_q = dist, q

            q_attn = attn[best_q].numpy().reshape(feat_h, feat_w)
            q_norm = q_attn / (q_attn.max() + 1e-8)
            attn_up = np.array(
                PILImage.fromarray((q_norm * 255).astype(np.uint8))
                .resize((args.img_size, args.img_size), PILImage.BILINEAR)
            ) / 255.0

            axes[col].imshow(img_np)
            axes[col].imshow(attn_up, alpha=0.55, cmap="jet")
            axes[col].set_title(
                f"Det {col}  conf={scores_np[det_idx]:.2f}\n"
                f"Query {best_q} cross-attn "
                f"[{last_key.split('.')[-3] if '.' in last_key else last_key}]",
                fontsize=8
            )
            axes[col].axis("off")

        fname = Path(img_info["file_name"]).stem
        plt.suptitle(f"D-FINE-S Decoder Cross-Attention — {fname}", fontsize=10)
        plt.tight_layout()
        save_path = out_dir / f"dfine_attn_{fname}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        processed += 1
        print(f"  [{processed}/{args.n_images}] Saved: {save_path.name}")

    # Restore original MHA forwards
    for mod_id, (module, orig_fwd) in module_originals.items():
        module.forward = orig_fwd

    if processed == 0:
        print("  No images processed via cross-attention. Trying GradCAM fallback...")
        _dfine_gradcam_fallback(args, images, img_folder, out_dir)
    else:
        print(f"\nD-FINE cross-attention complete. Saved {processed} images to: {out_dir}")


def _dfine_gradcam_fallback(args, images, img_folder, out_dir):
    """
    Fallback: GradCAM-style on D-FINE's backbone last feature map.
    Uses pytorch-grad-cam if available, otherwise skips.
    """
    try:
        from pytorch_grad_cam import GradCAM  # type: ignore
        import sys
        sys.path.insert(0, str(args.dfine_repo.expanduser().resolve()))
        import torch
        import torchvision.transforms as T
        from PIL import Image
        from src.core import YAMLConfig  # type: ignore
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] grad-cam not installed. pip install grad-cam")
        return

    cfg = YAMLConfig(
        str(args.dfine_config.expanduser().resolve()),
        resume=str(args.dfine_weights.expanduser().resolve())
    )
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    ckpt  = torch.load(str(args.dfine_weights.expanduser().resolve()), map_location="cpu")
    state = ckpt.get("ema", {}).get("module", ckpt.get("model", ckpt))
    cfg.model.load_state_dict(state, strict=False)
    model = cfg.model.eval()

    # Target: last conv block in backbone (HGNetv2)
    target_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and "backbone" in name:
            target_layers.append(module)

    if not target_layers:
        print("  [skip] Could not find backbone conv layer")
        return

    target = [target_layers[-1]]
    transform = T.Compose([T.Resize((args.img_size, args.img_size)), T.ToTensor()])

    # D-FINE returns a dict {'pred_logits':..., 'pred_boxes':...},
    # NOT a plain tensor. Standard SumTarget breaks. This handles all formats.
    class DetectionSumTarget:
        def __call__(self, output):
            if isinstance(output, torch.Tensor):
                return output.sum()
            if isinstance(output, dict):
                # Prefer classification logits (most sensitive to detection)
                for key in ("pred_logits", "pred_scores", "enc_topk_logits"):
                    if key in output and isinstance(output[key], torch.Tensor):
                        return output[key].sum()
                # Fallback: sum all tensor values
                tensors = [v for v in output.values()
                           if isinstance(v, torch.Tensor)]
                if tensors:
                    return sum(t.sum() for t in tensors)
            if isinstance(output, (list, tuple)):
                tensors = [v for v in output if isinstance(v, torch.Tensor)]
                if tensors:
                    return tensors[0].sum()
            raise RuntimeError(
                f"DetectionSumTarget: cannot sum type={type(output).__name__}: "
                f"{repr(output)[:80]}"
            )

    class DFineModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            out = self.model(x)
            if isinstance(out, dict):
                return out.get("pred_logits", torch.zeros(1, 1, 1))
            return out

    wrapped_model = DFineModelWrapper(model)
    cam = GradCAM(model=wrapped_model, target_layers=target)
    processed = 0
    for img_info in images[:args.n_images]:
        img = Image.open(img_folder / img_info["file_name"]).convert("RGB")
        img_np = np.array(img.resize((args.img_size, args.img_size))) / 255.0
        tensor = transform(img).unsqueeze(0)
        try:
            grayscale = cam(input_tensor=tensor, targets=[DetectionSumTarget()])[0]
        except Exception as e:
            print(f"  [warn] GradCAM failed: {e}")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_np); axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(img_np); axes[1].imshow(grayscale, alpha=0.5, cmap="jet")
        axes[1].set_title("D-FINE GradCAM (backbone)\nClever Hans check")
        axes[1].axis("off")
        fname = Path(img_info["file_name"]).stem
        plt.tight_layout()
        plt.savefig(out_dir / f"dfine_gradcam_{fname}.png", dpi=150, bbox_inches="tight")
        plt.close()
        processed += 1

    print(f"  GradCAM fallback: saved {processed} images to {out_dir}")


def _dfine_eigencam_fallback(args, images, img_folder, out_dir):
    """
    EigenCAM on D-FINE's backbone last feature map.

    EigenCAM (Muhammad et al. IJCNN 2020) uses PCA of feature activations—no
    backward pass needed, no target function, works with any backbone.
    This is a valid Clever Hans check: shows which image regions the backbone
    activates for, regardless of the decoder architecture.
    """
    try:
        from pytorch_grad_cam import EigenCAM  # type: ignore
        from pytorch_grad_cam.utils.image import show_cam_on_image  # type: ignore
    except ImportError:
        print("  [skip] grad-cam not installed: pip install grad-cam")
        return

    try:
        import sys
        sys.path.insert(0, str(args.dfine_repo.expanduser().resolve()))
        import torch
        import torchvision.transforms as T
        from PIL import Image as PILImage
        from src.core import YAMLConfig  # type: ignore
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as e:
        print(f"  [skip] missing dependency: {e}")
        return

    cfg = YAMLConfig(
        str(args.dfine_config.expanduser().resolve()),
        resume=str(args.dfine_weights.expanduser().resolve())
    )
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    ckpt  = torch.load(str(args.dfine_weights.expanduser().resolve()), map_location="cpu")
    state = ckpt.get("ema", {}).get("module", ckpt.get("model", ckpt))
    cfg.model.load_state_dict(state, strict=False)
    model = cfg.model.eval()  # CPU is fine for EigenCAM (no backward)

    # Target: last Conv2d in backbone (HGNetv2 stage4)
    backbone_convs = [(n, m) for n, m in model.named_modules()
                      if isinstance(m, torch.nn.Conv2d) and "backbone" in n]
    if not backbone_convs:
        print("  [skip] No backbone Conv2d found.")
        return
    target_layer = [backbone_convs[-1][1]]
    print(f"  EigenCAM target layer: {backbone_convs[-1][0]}")

    # Load deploy model separately for predictions
    cfg2 = YAMLConfig(
        str(args.dfine_config.expanduser().resolve()),
        resume=str(args.dfine_weights.expanduser().resolve())
    )
    if "HGNetv2" in cfg2.yaml_cfg:
        cfg2.yaml_cfg["HGNetv2"]["pretrained"] = False
    ckpt2 = torch.load(str(args.dfine_weights.expanduser().resolve()), map_location="cpu")
    state2 = ckpt2.get("ema", {}).get("module", ckpt2.get("model", ckpt2))
    cfg2.model.load_state_dict(state2, strict=False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    postproc2 = cfg2.postprocessor.deploy().to(device)
    model_deploy2 = cfg2.model.deploy().to(device).eval()

    class DFineModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            out = self.model(x)
            if isinstance(out, dict):
                return out.get("pred_logits", torch.zeros(1, 1, 1))
            return out

    wrapped_model = DFineModelWrapper(model)
    cam = EigenCAM(model=wrapped_model, target_layers=target_layer)

    transform = T.Compose([T.Resize((args.img_size, args.img_size)), T.ToTensor()])
    processed = 0

    for img_info in images:
        if processed >= args.n_images:
            break
        img_path = img_folder / img_info["file_name"]
        if not img_path.exists():
            continue

        img = PILImage.open(img_path).convert("RGB")
        w, h = img.size
        img_np = np.array(img.resize((args.img_size, args.img_size))) / 255.0
        tensor_cpu = transform(img).unsqueeze(0)          # on CPU for EigenCAM
        tensor_gpu = tensor_cpu.to(device)                # on GPU for predictions
        orig_size  = torch.tensor([[w, h]], device=device)

        with torch.no_grad():
            labels, boxes, scores = postproc2(model_deploy2(tensor_gpu), orig_size)
        scores_np = scores[0].cpu().numpy()
        boxes_np  = boxes[0].cpu().numpy()
        high_conf = scores_np >= args.min_conf

        if not high_conf.any():
            print(f"  [skip] No detections >= {args.min_conf:.2f} in {img_info['file_name']}")
            continue

        try:
            grayscale_cam = cam(input_tensor=tensor_cpu)[0]   # [H, W]
        except Exception as e:
            print(f"  [warn] EigenCAM failed for {img_info['file_name']}: {e}")
            continue

        cam_image = show_cam_on_image(
            img_np.astype(np.float32), grayscale_cam, use_rgb=True
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img_np)
        axes[0].set_title("Input + D-FINE Detections", fontsize=9)
        axes[0].axis("off")
        for box, conf in zip(boxes_np[high_conf], scores_np[high_conf]):
            x1 = int(box[0] * args.img_size / w)
            y1 = int(box[1] * args.img_size / h)
            x2 = int(box[2] * args.img_size / w)
            y2 = int(box[3] * args.img_size / h)
            rect = mpatches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor="lime", facecolor="none"
            )
            axes[0].add_patch(rect)
            axes[0].text(x1, y1-3, f"{conf:.2f}",
                         color="lime", fontsize=8, fontweight="bold")

        axes[1].imshow(cam_image)
        axes[1].set_title(
            "D-FINE-S EigenCAM (HGNetv2 backbone)\n"
            "Clever Hans check: red = high backbone activation",
            fontsize=9
        )
        axes[1].axis("off")

        fname = Path(img_info["file_name"]).stem
        plt.suptitle(f"D-FINE-S Backbone Saliency — {fname}", fontsize=10)
        plt.tight_layout()
        save_path = out_dir / f"dfine_eigencam_{fname}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        processed += 1
        print(f"  [{processed}/{args.n_images}] Saved: {save_path.name}")

    print(f"\nD-FINE EigenCAM complete. Saved {processed} images to: {out_dir}")


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = args.dataset_dir.expanduser().resolve()
    ann_file = dataset_dir / "annotations" / f"instances_{args.split}.json"
    images   = json.loads(ann_file.read_text())["images"]
    img_folder = dataset_dir / "images" / args.split

    if args.mode in ("eigen_cam", "both"):
        if args.yolo_weights and args.yolo_weights.expanduser().resolve().exists():
            print("\n=== EigenCAM on YOLO detection head ===")
            run_eigen_cam_yolo(args, images, img_folder)
        else:
            print("[skip] --yolo-weights not provided or not found")

    if args.mode in ("dfine_attn", "both"):
        if (args.dfine_config and args.dfine_weights and
                args.dfine_config.expanduser().resolve().exists() and
                args.dfine_weights.expanduser().resolve().exists()):
            print("\n=== D-FINE Decoder Cross-Attention Maps ===")
            run_dfine_cross_attention(args, images, img_folder)
        else:
            print("[skip] --dfine-config / --dfine-weights not provided or not found")


if __name__ == "__main__":
    main()
