"""
benchmark_dfine_cpu.py
----------------------
Controlled CPU inference speed benchmark for D-FINE-S.
Runs 10 warm-up + 50 timed inferences on CPU using a dummy tensor to measure latency and FPS.

Usage (in dfine_env):
  python benchmark_dfine_cpu.py
"""

import sys
import os
import time
import torch

BASE_DIR = os.environ.get("ITOBOS_ROOT", "/path/to/itobos")

DFINE_PATH = os.path.join(BASE_DIR, "aritra/dfine_code/D-FINE")
sys.path.insert(0, DFINE_PATH)

try:
    from src.core import YAMLConfig
except ImportError:
    print(f"Error: Could not import 'src.core' from {DFINE_PATH}.")
    print("Please check the path and ensure you run this script inside the 'dfine_env' environment.")
    exit(1)

config_path = os.path.join(BASE_DIR, "yamin/experiment-dfine/results/configs/dfine_s_fullaug_sweep_100ep.yml")
weight_path = os.path.join(BASE_DIR, "yamin/experiment-dfine/results/training/full/dfine_s_fullaug_sweep_100ep/best_stg2.pth")

IMGSZ  = 1024
WARMUP = 10
RUNS   = 50

def main():
    print("=" * 60)
    print("  CPU Inference Benchmark — D-FINE-S")
    print(f"  Config:  {config_path}")
    print(f"  Weights: {weight_path}")
    print("=" * 60)

    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        return
    if not os.path.exists(weight_path):
        print(f"Error: Weights file not found: {weight_path}")
        return

    cfg = YAMLConfig(config_path)
    checkpoint = torch.load(weight_path, map_location="cpu")

    state_dict = checkpoint.get("model", checkpoint)
    cfg.model.load_state_dict(state_dict)

    model = cfg.model.deploy().eval().to("cpu")

    dummy = torch.zeros(1, 3, IMGSZ, IMGSZ)

    print(f"Warming up ({WARMUP} runs)...", end=" ", flush=True)
    for _ in range(WARMUP):
        with torch.no_grad():
            _ = model(dummy)
    print("done")

    print(f"Benchmarking ({RUNS} runs)...", end=" ", flush=True)
    latencies = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy)
        latencies.append((time.perf_counter() - t0) * 1000)
    print("done")

    lat_mean = sum(latencies) / len(latencies)
    fps = 1000.0 / lat_mean

    summary_lines = []
    summary_lines.append("\n" + "=" * 60)
    summary_lines.append("  RESULTS")
    summary_lines.append("=" * 60)
    summary_lines.append(f"  Model:            D-FINE-S")
    summary_lines.append(f"  CPU Latency:      {lat_mean:.1f} ms")
    summary_lines.append(f"  CPU FPS:          {fps:.1f}")
    summary_lines.append("=" * 60)

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    output_txt = "benchmark_dfine_cpu_results.txt"
    try:
        with open(output_txt, "w") as f:
            f.write(summary_text + "\n")
        print(f"Results saved to: {os.path.abspath(output_txt)}")
    except Exception as e:
        print(f"Error saving to txt file: {e}")

if __name__ == "__main__":
    main()
