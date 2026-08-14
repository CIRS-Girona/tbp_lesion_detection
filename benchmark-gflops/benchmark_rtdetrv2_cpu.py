"""
benchmark_rtdetrv2_cpu.py
-------------------------
Controlled CPU inference speed benchmark for RT-DETRv2-S models (baseline and +SCALE).
Runs 10 warm-up + 50 timed inferences on CPU using a dummy tensor to measure latency and FPS.

Usage on server:
  python benchmark_rtdetrv2_cpu.py
"""

import sys
import os
import glob
import time
import torch

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

IMGSZ  = 1024
WARMUP = 10
RUNS   = 50

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

    # Look for config files containing 'r18' or 'resnet18'
    for fpath in all_configs:
        fname = os.path.basename(fpath).lower()
        if "r18" in fname or "resnet18" in fname:
            return fpath

    # Fallback to any rtdetrv2 config
    for fpath in all_configs:
        if "rtdetrv2" in fpath.lower():
            return fpath

    return None

def benchmark_rtdetrv2(name, info):
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {name}")
    print(f"{'='*60}")

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

    dummy = torch.zeros(1, 3, IMGSZ, IMGSZ)

    print(f"  Warming up ({WARMUP} runs)...", end=" ", flush=True)
    for _ in range(WARMUP):
        with torch.no_grad():
            _ = model(dummy)
    print("done")

    print(f"  Benchmarking ({RUNS} runs)...", end=" ", flush=True)
    latencies = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy)
        latencies.append((time.perf_counter() - t0) * 1000)
    print("done")

    lat_mean = sum(latencies) / len(latencies)
    fps = 1000.0 / lat_mean

    print(f"  → Latency: {lat_mean:.1f} ms")
    print(f"  → FPS:     {fps:.1f}")

    return {
        "model": name,
        "latency_ms": round(lat_mean, 1),
        "fps": round(fps, 1)
    }

def main():
    results = {}
    for name, info in models_info.items():
        res = benchmark_rtdetrv2(name, info)
        if res:
            results[name] = res

    summary_lines = []
    summary_lines.append("\n" + "=" * 60)
    summary_lines.append("  SUMMARY TABLE")
    summary_lines.append("=" * 60)
    summary_lines.append(f"  {'Model':<25} | {'CPU Latency (ms)':<18} | {'CPU FPS':<10}")
    summary_lines.append("-" * 60)
    for name, res in results.items():
        summary_lines.append(f"  {name:<25} | {res['latency_ms']:>18.1f} | {res['fps']:>10.1f}")
    summary_lines.append("=" * 60)

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    output_txt = "benchmark_rtdetrv2_cpu_results.txt"
    try:
        with open(output_txt, "w") as f:
            f.write(summary_text + "\n")
        print(f"Results saved to: {os.path.abspath(output_txt)}")
    except Exception as e:
        print(f"Error saving to txt file: {e}")

if __name__ == "__main__":
    main()
