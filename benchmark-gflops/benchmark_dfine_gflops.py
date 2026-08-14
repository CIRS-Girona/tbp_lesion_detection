"""
benchmark_dfine_gflops.py
-------------------------
Computes GFLOPs (thop MACs) at 1024x1024 for D-FINE-S on the server.
Note: 640px is skipped because the model's positional embeddings are hardcoded/fixed for 1024px.

Usage (in dfine_env):
  python benchmark_dfine_gflops.py
"""

import sys
import os
import torch

try:
    import thop
except ImportError:
    print("Error: 'thop' package is not installed. Please run: pip install thop")
    exit(1)

BASE_DIR = os.environ.get("ITOBOS_ROOT", "/path/to/itobos")

DFINE_PATH = os.path.join(BASE_DIR, "aritra/dfine_code/D-FINE")
sys.path.insert(0, DFINE_PATH)

try:
    from src.core import YAMLConfig
except ImportError:
    print(f"Error: Could not import 'src.core' from {DFINE_PATH}.")
    exit(1)

config_path = os.path.join(BASE_DIR, "yamin/experiment-dfine/results/configs/dfine_s_fullaug_sweep_100ep.yml")
weight_path = os.path.join(BASE_DIR, "yamin/experiment-dfine/results/training/full/dfine_s_fullaug_sweep_100ep/best_stg2.pth")

def main():
    print("=" * 60)
    print("  GFLOPs Benchmark — D-FINE-S")
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

    # 1024px only (640px is skipped because of positional embedding size mismatch)
    dummy_1024 = torch.zeros(1, 3, 1024, 1024)
    macs_1024, _ = thop.profile(model, inputs=(dummy_1024,), verbose=False)
    g_1024 = macs_1024 / 1e9

    print("\n" + "=" * 60)
    print("  RESULTS (thop MACs GFLOPs)")
    print("=" * 60)
    print(f"  Model:            D-FINE-S")
    print(f"  GFLOPs @ 1024px:  {g_1024:.2f} G")
    print("=" * 60)

if __name__ == "__main__":
    main()
