#!/usr/bin/env python3
"""
compute_model_sizes_gflops.py
==============================
Run this on the server to get:
  1) Deployment-only model size (MB) for D-FINE-S and RT-DETRv2-S
  2) GFLOPs for D-FINE-S at 1024px

Usage (on server, from any folder):
    source ~/miniconda3/bin/activate   (or your conda/venv)
    python compute_model_sizes_gflops.py

Output: prints a ready-to-paste table row for the paper.
"""

import os, sys
import torch

DFINE_WEIGHTS  = os.environ.get(
    "DFINE_WEIGHTS",
    os.path.expanduser("~/code/iToBoS/yamin/experiment-dfine/results/training/full"
                       "/dfine_s_fullaug_sweep_100ep/best_stg2.pth")
)
DFINE_CONFIG   = os.environ.get(
    "DFINE_CONFIG",
    os.path.expanduser("~/code/iToBoS/yamin/experiment-dfine/results/configs"
                       "/dfine_s_fullaug_sweep_100ep.yml")
)
DFINE_REPO     = os.environ.get(
    "DFINE_REPO",
    os.path.expanduser("~/code/iToBoS/aritra/dfine_code/D-FINE")
)

RTDETR_WEIGHTS = os.environ.get(
    "RTDETR_WEIGHTS",
    os.path.expanduser("~/code/iToBoS/praveen/rtdetrv2_output/rtdetrv2_r18vd_itobos_besthp_100e_1024/best.pth")
)
# Fallback search paths for the RT-DETRv2-S checkpoint
RTDETR_FALLBACK_PATHS = [
    "~/code/iToBoS/praveen/best_baseline.pth",
    "~/code/iToBoS/praveen/rtdetr/best_baseline.pth",
    "~/code/iToBoS/praveen/experiment-rtdetr/runs/best_baseline.pth",
]


def find_file(primary, fallbacks):
    p = os.path.expanduser(primary)
    if os.path.exists(p):
        return p
    for f in fallbacks:
        fp = os.path.expanduser(f)
        if os.path.exists(fp):
            return fp
    return None


def deployment_size_mb(ckpt_path):
    """
    Load checkpoint, extract only model weights (no optimizer/EMA),
    save to a temp buffer, return size in MB.
    This is what 'deployment-only weights' means.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "model" in ckpt:
        state = ckpt["model"]
    elif "ema" in ckpt and "module" in ckpt["ema"]:
        state = ckpt["ema"]["module"]      # EMA weights = what you'd deploy
    else:
        state = ckpt                       # already weights-only

    total_bytes = sum(v.numel() * v.element_size() for v in state.values()
                      if isinstance(v, torch.Tensor))
    return round(total_bytes / 1e6, 1)


def dfine_gflops_1024(config_path, weights_path, repo_path, img_size=1024):
    """
    Compute D-FINE-S GFLOPs (MACs) at img_size x img_size.
    Uses fvcore if available, falls back to thop, falls back to manual count.
    """
    sys.path.insert(0, repo_path)
    from src.core import YAMLConfig  # type: ignore

    cfg = YAMLConfig(config_path, resume=weights_path)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    ckpt  = torch.load(weights_path, map_location="cpu")
    state = ckpt.get("ema", {}).get("module", ckpt.get("model", ckpt))
    cfg.model.load_state_dict(state, strict=False)
    model = cfg.model.eval()

    dummy = torch.randn(1, 3, img_size, img_size)

    # Try fvcore first (handles non-standard forward signatures better)
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, dummy)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        gflops = round(flops.total() / 1e9, 1)
        return gflops, "fvcore"
    except Exception as e:
        print(f"  [fvcore] failed: {e}")

    # Try thop
    try:
        from thop import profile  # type: ignore
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return round(macs / 1e9, 1), "thop-MACs"
    except Exception as e:
        print(f"  [thop] failed: {e}")

    # Manual: count only backbone Conv2d (gives backbone MACs as lower bound)
    total_macs = 0
    x = dummy
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and "backbone" in name:
            # MACs for a conv layer = Cin * Cout * kH * kW * outH * outW
            kH, kW = module.kernel_size
            outH = img_size // (2 ** name.count("stage"))  # approx
            outW = outH
            macs = module.in_channels * module.out_channels * kH * kW * outH * outW
            total_macs += macs
    return round(total_macs / 1e9, 1), "manual-backbone-only"


def main():
    print("=" * 60)
    print("MODEL SIZE + GFLOPs COMPUTATION")
    print("=" * 60)

    print("\n--- D-FINE-S ---")
    if not os.path.exists(DFINE_WEIGHTS):
        print(f"[ERROR] Weights not found: {DFINE_WEIGHTS}")
        dfine_size = None
    else:
        print(f"Checkpoint: {DFINE_WEIGHTS}")
        ckpt_mb    = round(os.path.getsize(DFINE_WEIGHTS) / 1e6, 1)
        deploy_mb  = deployment_size_mb(DFINE_WEIGHTS)
        print(f"  Full checkpoint size : {ckpt_mb} MB")
        print(f"  Deployment-only size : {deploy_mb} MB")
        dfine_size = deploy_mb

    print("\nComputing D-FINE-S GFLOPs at 1024px...")
    try:
        gflops, method = dfine_gflops_1024(DFINE_CONFIG, DFINE_WEIGHTS, DFINE_REPO)
        print(f"  GFLOPs @ 1024px : {gflops}  (method: {method})")
    except Exception as e:
        print(f"  [ERROR] GFLOPs computation failed: {e}")
        gflops = None

    print("\n--- RT-DETRv2-S ---")
    rtdetr_path = find_file(RTDETR_WEIGHTS, RTDETR_FALLBACK_PATHS)
    if rtdetr_path is None:
        print(f"[WARN] RT-DETRv2-S checkpoint not found at any known path.")
        print("       Run this command manually to find it:")
        print("       find ~/code/iToBoS/praveen -name 'best_baseline.pth' 2>/dev/null")
        rtdetr_size = None
    else:
        print(f"Checkpoint: {rtdetr_path}")
        ckpt_mb   = round(os.path.getsize(rtdetr_path) / 1e6, 1)
        deploy_mb = deployment_size_mb(rtdetr_path)
        print(f"  Full checkpoint size : {ckpt_mb} MB")
        print(f"  Deployment-only size : {deploy_mb} MB")
        rtdetr_size = deploy_mb

    print("\n" + "=" * 60)
    print("COPY-PASTE SUMMARY FOR PAPER TABLE:")
    print("=" * 60)
    print(f"  D-FINE-S    | Size = {dfine_size or '???'} MB | GFLOPs @ 1024px = {gflops or '???'}")
    print(f"  RT-DETRv2-S | Size = {rtdetr_size or '???'} MB | GFLOPs @ 1024px = 76.0 (already correct)")
    print("=" * 60)


if __name__ == "__main__":
    main()
