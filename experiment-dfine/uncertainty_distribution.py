#!/usr/bin/env python3
"""
D-FINE Localization Uncertainty Distribution Plot
+ YOLO DFL Uncertainty Recovery

Hayat's email:
  "First, by plotting the localization uncertainty as I mentioned in the meeting.
   D-FINE does this by default (see the paper for examples). For YOLO, you should
   be able to recover this from DFL."

For D-FINE:
  The FGD head outputs a softmax distribution p_k over K=17 bins per coordinate.
  We extract these raw distributions and visualize:
    - For selected test images: the 4 distributions (x1,y1,x2,y2) per detected box
    - Global histogram: entropy of all distributions (low entropy = peaked = confident)
    - Scatter: entropy vs IoU with GT (should be negatively correlated)

For YOLO:
  DFL (Distribution Focal Loss) outputs a distribution over L=16 bins
  per coordinate side (left/top/right/bottom). We extract via hooks
  and compute the same uncertainty metrics.

Usage:
  python scripts/uncertainty_distribution.py \
    --dfine-config results/configs/dfine_s_fullaug_sweep_100ep.yml \
    --dfine-weights results/training/full/dfine_s_fullaug_sweep_100ep/best_stg2.pth \
    --yolo-weights /path/to/yolov12s.pt \
    --dataset-dir /path/to/dataset_coco \
    --split test \
    --n-images 20
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

DFINE_REPO  = Path(os.environ.get("DFINE_REPO", "~/D-FINE")).expanduser()
DATASET_DIR = Path(os.environ.get("DFINE_DATASET_DIR", "~/dfine_results/dataset_coco")).expanduser()
RESULTS_DIR = Path(os.environ.get("DFINE_RESULTS_DIR", "results")).expanduser()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dfine-repo",     type=Path, default=DFINE_REPO)
    p.add_argument("--dataset-dir",    type=Path, default=DATASET_DIR)
    p.add_argument("--output-dir",     type=Path,
                   default=RESULTS_DIR / "uncertainty_analysis")
    p.add_argument("--dfine-config",   type=Path, default=None)
    p.add_argument("--dfine-weights",  type=Path, default=None)
    p.add_argument("--yolo-weights",   type=Path, default=None,
                   help="YOLOv12s .pt weights for DFL uncertainty")
    p.add_argument("--split",          choices=["val","test"], default="test")
    p.add_argument("--img-size",       type=int, default=1024)
    p.add_argument("--device",         type=str, default="cuda:0")
    p.add_argument("--n-images",       type=int, default=20,
                   help="Number of sample images for qualitative plots")
    p.add_argument("--min-conf",       type=float, default=0.3,
                   help="Min confidence to include in distribution plots")
    return p.parse_args()


def entropy(probs: np.ndarray, eps: float = 1e-9) -> float:
    """Shannon entropy of a probability distribution."""
    return float(-np.sum(probs * np.log(probs + eps)))


def extract_dfine_distributions(args: argparse.Namespace,
                                  images: list) -> Dict:
    """
    Extract per-box coordinate distributions from D-FINE's decoder output.

    D-FINE FGD head: after the decoder, each box head outputs logits of shape
    [batch, num_queries, 4*K] where K=17 bins. We hook into the model to
    capture these raw logits before box decoding.
    """
    import sys
    sys.path.insert(0, str(args.dfine_repo.expanduser().resolve()))

    import torch
    import torchvision.transforms as T
    from PIL import Image
    from src.core import YAMLConfig  # type: ignore

    cfg = YAMLConfig(str(args.dfine_config.expanduser().resolve()),
                     resume=str(args.dfine_weights.expanduser().resolve()))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    ckpt = torch.load(str(args.dfine_weights.expanduser().resolve()), map_location="cpu")
    state = ckpt.get("ema", {}).get("module", ckpt.get("model", ckpt))
    cfg.model.load_state_dict(state, strict=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = cfg.model.to(device).eval()

    # Hook to capture raw distribution logits from the regression head
    captured_dists = []

    def hook_fn(module, input, output):
        # output shape: [B, num_queries, 4*K]
        with torch.no_grad():
            logits = output.detach().cpu()
            num_features = logits.shape[-1]
            if num_features % 4 == 0:
                local_K = num_features // 4
                probs = torch.softmax(logits.reshape(*logits.shape[:-1], 4, local_K), dim=-1)
                captured_dists.append(probs.numpy())

    transform = T.Compose([T.Resize((args.img_size, args.img_size)), T.ToTensor()])
    img_folder = args.dataset_dir.expanduser().resolve() / "images" / args.split

    all_entropies = []
    all_confs = []
    sample_dists = []  # For qualitative plots

    # Instantiate postprocessor and deploy model outside the loop
    # to prevent double-deploy crashes
    postproc = cfg.postprocessor.deploy().to(device)
    model_deploy = cfg.model.deploy().to(device)

    # Register hook on the deployed model's regression prediction head
    hooks = []
    for name, module in model_deploy.named_modules():
        if "dec_bbox_head" in name.lower() and name.endswith(".layers.2") and \
           isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        for idx, img_info in enumerate(images[:args.n_images * 5]):
            captured_dists.clear()
            img = Image.open(img_folder / img_info["file_name"]).convert("RGB")
            w, h = img.size
            tensor = transform(img).unsqueeze(0).to(device)
            orig_size = torch.tensor([[w, h]], device=device)

            labels, boxes, scores = postproc(model_deploy(tensor), orig_size)

            scores_np = scores[0].cpu().numpy()
            high_conf_mask = scores_np >= args.min_conf

            if high_conf_mask.any() and captured_dists:
                # dists shape: [B, num_queries, 4, K]
                dists = captured_dists[-1][0]  # [num_queries, 4, K]
                for q_idx in np.where(high_conf_mask)[0]:
                    if q_idx < len(dists):
                        box_dists = dists[q_idx]  # [4, K]
                        coord_entropies = [entropy(box_dists[c]) for c in range(4)]
                        all_entropies.append(np.mean(coord_entropies))
                        all_confs.append(float(scores_np[q_idx]))

                if len(sample_dists) < args.n_images:
                    # Pick the highest-confidence detection for qualitative plot
                    best_q = int(np.argmax(scores_np))
                    if best_q < len(dists):
                        sample_dists.append({
                            "file": img_info["file_name"],
                            "score": float(scores_np[best_q]),
                            "distributions": dists[best_q].tolist(),  # [4, K]
                        })

    for h in hooks:
        h.remove()

    return {
        "all_entropies": all_entropies,
        "all_confs": all_confs,
        "sample_dists": sample_dists,
    }


def extract_yolo_dfl_distributions(args: argparse.Namespace,
                                     images: list) -> Dict:
    """
    Extract DFL distributions from YOLO detection head.
    YOLO DFL uses 16 bins per coordinate side (left/top/right/bottom).
    """
    try:
        import sys
        import torch
        import torchvision.transforms as T
        from PIL import Image
        from ultralytics import YOLO  # type: ignore
    except ImportError as e:
        print(f"[skip] ultralytics not available: {e}")
        return {}

    model = YOLO(str(args.yolo_weights.expanduser().resolve()))
    L = 16  # YOLO DFL bins
    captured_dfl = []

    def dfl_hook(module, input, output):
        with torch.no_grad():
            # input[0] shape is [B, 4*L, Anchors] containing the raw logits
            t = input[0].detach().cpu()
            if t.ndim == 3 and t.shape[1] == 4 * L:
                B, _, Anchors = t.shape
                # Reshape to [B, 4, L, Anchors]
                t = t.reshape(B, 4, L, Anchors)
                # Softmax over the L dimension (bins)
                probs = torch.softmax(t, dim=2)
                # Permute to [B, Anchors, 4, L]
                probs = probs.permute(0, 3, 1, 2)
                captured_dfl.append(probs.numpy())

    hooks = []
    for name, module in model.model.named_modules():
        if "dfl" in name.lower() or "DFL" in type(module).__name__:
            hooks.append(module.register_forward_hook(dfl_hook))

    transform = T.Compose([T.Resize((args.img_size, args.img_size)), T.ToTensor()])
    img_folder = args.dataset_dir.expanduser().resolve() / "images" / args.split

    all_entropies = []
    all_confs = []

    with torch.no_grad():
        for img_info in images[:args.n_images * 5]:
            captured_dfl.clear()
            img = Image.open(img_folder / img_info["file_name"]).convert("RGB")
            results = model.predict(
                np.array(img), imgsz=args.img_size, device="cpu", verbose=False,
                conf=args.min_conf,
            )
            if results and len(results[0].boxes) > 0 and captured_dfl:
                confs = results[0].boxes.conf.cpu().numpy()
                for dfl_output in captured_dfl:
                    # dfl_output: [B, H, W, 4, L]
                    probs_flat = dfl_output[0]  # [Anchors, 4, L]
                    for anchor_probs in probs_flat[:len(confs)]:
                        coord_entropies = [entropy(anchor_probs[c]) for c in range(4)]
                        all_entropies.append(np.mean(coord_entropies))
                    all_confs.extend(confs[:min(len(confs), len(probs_flat))])

    for h in hooks:
        h.remove()

    return {"all_entropies": all_entropies, "all_confs": all_confs}


def plot_uncertainty_comparison(dfine_data: dict, yolo_data: dict, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not available")
        return

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # Entropy histograms: D-FINE vs YOLO
    ax1 = fig.add_subplot(gs[0, 0])
    if dfine_data.get("all_entropies"):
        ax1.hist(dfine_data["all_entropies"], bins=30, alpha=0.7,
                 label="D-FINE-S", color="royalblue", density=True)
    if yolo_data.get("all_entropies"):
        ax1.hist(yolo_data["all_entropies"], bins=30, alpha=0.7,
                 label="YOLOv12s", color="darkorange", density=True)
    ax1.set_xlabel("Distribution Entropy (lower = more confident)")
    ax1.set_ylabel("Density")
    ax1.set_title("Localization Uncertainty Distribution")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Conf vs Entropy scatter
    ax2 = fig.add_subplot(gs[0, 1])
    if dfine_data.get("all_entropies"):
        ax2.scatter(dfine_data["all_confs"], dfine_data["all_entropies"],
                    alpha=0.3, s=8, c="royalblue", label="D-FINE-S")
    if yolo_data.get("all_entropies"):
        ax2.scatter(yolo_data["all_confs"], yolo_data["all_entropies"],
                    alpha=0.3, s=8, c="darkorange", label="YOLOv12s")
    ax2.set_xlabel("Detection Confidence Score")
    ax2.set_ylabel("Distribution Entropy")
    ax2.set_title("Confidence vs Localization Uncertainty")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Sample D-FINE distributions (qualitative)
    ax3 = fig.add_subplot(gs[0, 2])
    coord_names = ["left", "top", "right", "bottom"]
    K = len(dfine_data["sample_dists"][0]["distributions"][0]) if dfine_data.get("sample_dists") else 17
    if dfine_data.get("sample_dists"):
        sample = dfine_data["sample_dists"][0]
        dists = np.array(sample["distributions"])  # [4, K]
        x = np.arange(K)
        colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]
        for c, (dist, name) in enumerate(zip(dists, coord_names)):
            ax3.plot(x, dist, label=name, color=colors[c], marker="o", markersize=3)
        ax3.set_title(f"D-FINE Box Distributions\nconf={sample['score']:.2f}")
        ax3.set_xlabel("Bin index (0–16)")
        ax3.set_ylabel("Probability")
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)

    # Mean entropy bar comparison
    ax4 = fig.add_subplot(gs[1, :2])
    means = {}
    if dfine_data.get("all_entropies"):
        means["D-FINE-S"] = np.mean(dfine_data["all_entropies"])
    if yolo_data.get("all_entropies"):
        means["YOLOv12s (DFL)"] = np.mean(yolo_data["all_entropies"])
    if means:
        bars = ax4.bar(list(means.keys()), list(means.values()),
                       color=["royalblue", "darkorange"][:len(means)])
        for bar, val in zip(bars, means.values()):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                     f"{val:.4f}", ha="center", fontsize=10)
        ax4.set_ylabel("Mean Entropy per Detection")
        ax4.set_title("Mean Localization Uncertainty — Lower is Better (More Confident)")
        ax4.grid(axis="y", alpha=0.3)

    plt.suptitle("Localization Uncertainty Analysis: D-FINE vs YOLO", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "uncertainty_comparison.png", dpi=200)
    plt.close()
    print(f"Saved: {out_dir}/uncertainty_comparison.png")


def plot_sample_dfine_dists(sample_dists: list, out_dir: Path, n: int = 6) -> None:
    """Plot coordinate distributions for n sample detections — for the paper."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    K = len(sample_dists[0]["distributions"][0]) if sample_dists else 17
    coord_names = ["$\\hat{x}_1$", "$\\hat{y}_1$", "$\\hat{x}_2$", "$\\hat{y}_2$"]
    colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]

    fig, axes = plt.subplots(min(n, len(sample_dists)), 4,
                              figsize=(16, 3 * min(n, len(sample_dists))))
    if min(n, len(sample_dists)) == 1:
        axes = [axes]

    for row, (ax_row, sample) in enumerate(zip(axes, sample_dists[:n])):
        dists = np.array(sample["distributions"])  # [4, K]
        for col, (ax, dist, name, color) in enumerate(
                zip(ax_row, dists, coord_names, colors)):
            ax.bar(np.arange(K), dist, color=color, alpha=0.75)
            ax.set_title(f"Row {row+1} · {name}\nconf={sample['score']:.2f}", fontsize=8)
            ax.set_xlabel("Bin", fontsize=7)
            ax.set_ylabel("P", fontsize=7)
            ax.set_xticks(np.arange(0, K, 4))
            peak_entropy = entropy(dist)
            ax.set_title(f"{name}  H={peak_entropy:.2f}", fontsize=8)

    plt.suptitle("D-FINE-S: FGD Probability Distributions per Box Coordinate\n"
                 "(peaked distribution = high localization confidence)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "dfine_distribution_samples.png", dpi=200)
    plt.close()
    print(f"Saved: {out_dir}/dfine_distribution_samples.png")


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = args.dataset_dir.expanduser().resolve()
    ann_file = dataset_dir / "annotations" / f"instances_{args.split}.json"
    images = json.loads(ann_file.read_text())["images"]

    dfine_data = {}
    yolo_data  = {}

    if args.dfine_config and args.dfine_weights:
        cfg_path = args.dfine_config.expanduser().resolve()
        wt_path  = args.dfine_weights.expanduser().resolve()
        if cfg_path.exists() and wt_path.exists():
            print("=== Extracting D-FINE uncertainty distributions ===")
            dfine_data = extract_dfine_distributions(args, images)
            json.dumps(dfine_data, default=str)  # validate serializable

    if args.yolo_weights:
        wt_path = args.yolo_weights.expanduser().resolve()
        if wt_path.exists():
            print("=== Extracting YOLO DFL distributions ===")
            yolo_data = extract_yolo_dfl_distributions(args, images)

    if dfine_data.get("sample_dists"):
        plot_sample_dfine_dists(dfine_data["sample_dists"], out_dir)

    if dfine_data or yolo_data:
        plot_uncertainty_comparison(dfine_data, yolo_data, out_dir)

    # Save entropy statistics
    stats = {}
    if dfine_data.get("all_entropies"):
        h = np.array(dfine_data["all_entropies"])
        stats["D-FINE-S"] = {
            "mean_entropy": float(np.mean(h)),
            "std_entropy":  float(np.std(h)),
            "n_detections": len(h),
        }
    if yolo_data.get("all_entropies"):
        h = np.array(yolo_data["all_entropies"])
        stats["YOLOv12s"] = {
            "mean_entropy": float(np.mean(h)),
            "std_entropy":  float(np.std(h)),
            "n_detections": len(h),
        }
    (out_dir / "entropy_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nEntropy statistics:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
