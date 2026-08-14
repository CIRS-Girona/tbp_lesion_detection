"""
benchmark_rtdetrv2_gflops.py
----------------------------
Computes GFLOPs (thop MACs) at 1024x1024 for RT-DETRv2-S models (baseline and +SCALE) on the server.
Note: 640px is skipped because the model's positional embeddings are hardcoded/fixed for 1024px.

Usage on server:
  python benchmark_rtdetrv2_gflops.py
"""

import sys
import os
import glob
import torch

try:
    import thop
except ImportError:
    print("Error: 'thop' package is not installed. Please run: pip install thop")
    exit(1)

BASE_DIR = os.environ.get("ITOBOS_ROOT", "/path/to/itobos")

def find_codebase_dir():
    """Locate the RT-DETRv2 codebase directory containing 'src/core/yaml_config.py'."""
    search_dirs = [
        os.path.join(BASE_DIR, "praveen/rtdetrv2/rtdetrv2_pytorch"),
        os.path.join(BASE_DIR, "praveen/RT-DETR/rtdetrv2_pytorch"),
        os.path.join(BASE_DIR, "praveen/RT-DETR"),
        os.path.join(BASE_DIR, "praveen"),
    ]
    for d in search_dirs:
        if os.path.exists(os.path.join(d, "src/core/yaml_config.py")):
            return d
    # Recursive search as a fallback
    praveen_root = os.path.join(BASE_DIR, "praveen")
    if os.path.exists(praveen_root):
        for root, dirs, files in os.walk(praveen_root):
            if "yaml_config.py" in files and root.endswith("src/core"):
                return os.path.dirname(os.path.dirname(root))
    return None

codebase_dir = find_codebase_dir()
if not codebase_dir:
    print("Error: Could not locate the RT-DETRv2 codebase directory containing 'src/core/yaml_config.py'.")
    exit(1)

print(f"Located RT-DETRv2 codebase: {codebase_dir}")
sys.path.insert(0, codebase_dir)

from src.core import YAMLConfig

models_info = {
    "RT-DETRv2-S Baseline": {
        "dir": os.path.join(BASE_DIR, "praveen/rtdetrv2_output/rtdetrv2_r18vd_itobos_besthp_100e_1024"),
    },
    "RT-DETRv2-S + SCALE": {
        "dir": os.path.join(BASE_DIR, "praveen/rtdetrv2_output/rtdetrv2_r18vd_itobos_scale_besthp_100e_1024"),
    }
}

def find_config_file(model_name):
    """Search recursively for a matching config file."""
    praveen_configs_dir = os.path.join(codebase_dir, "configs")
    search_dirs = [
        praveen_configs_dir,
        os.path.join(BASE_DIR, "praveen"),
    ]

    all_configs = []
    for s_dir in search_dirs:
        if not os.path.exists(s_dir):
            continue
        for root, dirs, files in os.walk(s_dir):
            for f in files:
                if f.endswith((".yml", ".yaml")):
                    all_configs.append(os.path.join(root, f))

    is_scale = "scale" in model_name.lower()

    # Look for config files containing 'itobos' and 'r18' with scale matching
    for fpath in all_configs:
        fname = os.path.basename(fpath).lower()
        if "itobos" in fname and "r18" in fname:
            if is_scale and "scale" in fname:
                return fpath
            if not is_scale and "scale" not in fname:
                return fpath

    # Look for config files containing 'rtdetrv2_r18vd' with scale matching
    for fpath in all_configs:
        fname = os.path.basename(fpath).lower()
        if "rtdetrv2_r18vd" in fname:
            if is_scale and "scale" in fname:
                return fpath
            if not is_scale and "scale" not in fname:
                return fpath

    return None

def benchmark_rtdetrv2_gflops(name, info):
    print(f"\nBenchmarking GFLOPs for: {name}")
    model_dir = info["dir"]
    if not os.path.exists(model_dir):
        print(f"  [SKIP] Model directory not found: {model_dir}")
        return None

    weights_path = os.path.join(model_dir, "best.pth")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(model_dir, "checkpoint.pth")
        if not os.path.exists(weights_path):
            print(f"  [SKIP] Weights file not found in {model_dir}")
            return None

    config_path = find_config_file(name)
    if not config_path:
        print(f"  [SKIP] Config file (.yml/.yaml) for model '{name}' could not be located.")
        return None

    print(f"  Config:  {config_path}")
    print(f"  Weights: {weights_path}")

    try:
        cfg = YAMLConfig(config_path)
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        cfg.model.load_state_dict(state_dict)

        if hasattr(cfg.model, "deploy"):
            model = cfg.model.deploy().eval().to("cpu")
        else:
            model = cfg.model.eval().to("cpu")
    except Exception as e:
        print(f"  [ERROR] Failed to load model: {e}")
        return None

    # 1024px only (640px is skipped because of positional embedding size mismatch)
    dummy_1024 = torch.zeros(1, 3, 1024, 1024)
    macs_1024, _ = thop.profile(model, inputs=(dummy_1024,), verbose=False)
    g_1024 = macs_1024 / 1e9

    print(f"  {name}: {g_1024:.2f} GFLOPs @ 1024px")
    return {"1024": g_1024}

def main():
    print("=" * 70)
    print("  GFLOPs Benchmark (RT-DETRv2-S baseline and +SCALE)")
    print("  Device: CPU | Convention: thop MACs")
    print("=" * 70)

    results = {}
    for name, info in models_info.items():
        res = benchmark_rtdetrv2_gflops(name, info)
        if res:
            results[name] = res

    print("\n" + "=" * 70)
    print("  SUMMARY TABLE (thop MACs GFLOPs)")
    print("=" * 70)
    print(f"  {'Model':<25} | {'GFLOPs @ 1024px':<16}")
    print("-" * 70)
    for name, res in results.items():
        print(f"  {name:<25} | {res['1024']:>16.2f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
