#!/usr/bin/env python3
"""
gpu_benchmark_yolo.py
======================
Dedicated GPU benchmark for ALL YOLO models using the same protocol
as RT-DETRv2-S and D-FINE-S:
  - 20 warmup runs (no timing)
  - 100 timed runs
  - Dummy 1x3x1024x1024 input tensor (no preprocessing overhead)
  - torch.cuda.synchronize() around each run

Run on server:
    source ~/miniconda3/bin/activate   (or your conda/venv)
    python gpu_benchmark_yolo.py

Output: ready-to-paste table rows for main.tex
"""

import os
import time
import torch
from pathlib import Path

BASE = Path(os.environ.get("YOLO_EXPERIMENTS_DIR", "~/code/iToBoS/yamin")).expanduser()

MODELS = {
    "YOLOv8s":  BASE / "experiment-v8/runs/best_model/yolo8s_best_full_100ep/weights/best.pt",
    "YOLOv11s": BASE / "experiment-v11/runs/best_model/yolo11s_best_full_100ep/weights/best.pt",
    "YOLOv12s": BASE / "experiment-v12/runs/best_model/yolo12s_best_full_100ep/weights/best.pt",
    "YOLOv26s": BASE / "experiment-v26/runs/best_model/yolo26s_best_full_100ep/weights/best.pt",
}

# Fallback: glob for any name variation (e.g. yolov8s instead of yolo8s)
FALLBACKS = {
    "YOLOv8s":  "experiment-v8/runs/best_model/*/weights/best.pt",
    "YOLOv11s": "experiment-v11/runs/best_model/*/weights/best.pt",
    "YOLOv12s": "experiment-v12/runs/best_model/*/weights/best.pt",
    "YOLOv26s": "experiment-v26/runs/best_model/*/weights/best.pt",
}

WARMUP = 20
RUNS   = 100
IMGSZ  = 1024
DEVICE = "cuda:0"   # change to cuda:1/2/3 if needed


def find_weight(name):
    primary = MODELS[name]
    if primary.exists():
        return primary
    # Try glob fallback
    pattern = FALLBACKS[name]
    matches = sorted(BASE.glob(pattern))
    # Prefer full-aug (skip noaug)
    full_matches = [m for m in matches if "noaug" not in str(m).lower()]
    if full_matches:
        return full_matches[0]
    if matches:
        return matches[0]
    return None


def benchmark_yolo_gpu(weight_path, device, warmup=20, runs=100, imgsz=1024):
    """
    Pure forward-pass GPU benchmark.
    Uses the underlying PyTorch model directly (no ultralytics preprocessing).
    Same protocol as RT-DETRv2-S: dummy input, synchronize, timed loop.
    """
    from ultralytics import YOLO

    model = YOLO(str(weight_path))
    torch_model = model.model.to(device).eval()

    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = torch_model(dummy)
            torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = torch_model(dummy)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)  # ms

    mean_ms = sum(times) / len(times)
    fps     = 1000.0 / mean_ms
    return round(mean_ms, 1), round(fps, 1)


def main():
    if not torch.cuda.is_available():
        print("[ERROR] No GPU found. This script must run on the server with CUDA.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Protocol: {WARMUP} warmup + {RUNS} timed runs, "
          f"dummy {IMGSZ}x{IMGSZ} input, forward-pass only\n")

    results = {}
    for name in ["YOLOv8s", "YOLOv11s", "YOLOv12s", "YOLOv26s"]:
        w = find_weight(name)
        if w is None:
            print(f"[SKIP] {name}: weights not found")
            results[name] = None
            continue
        print(f"[{name}] Loading {w.name} ...")
        try:
            lat_ms, fps = benchmark_yolo_gpu(w, DEVICE, WARMUP, RUNS, IMGSZ)
            results[name] = (lat_ms, fps)
            print(f"  → {lat_ms} ms  |  {fps} FPS")
        except Exception as e:
            print(f"  [ERROR] {e}")
            results[name] = None

    print("\n" + "=" * 60)
    print("COPY-PASTE FOR main.tex (GPU Lat. and GPU FPS columns):")
    print("=" * 60)
    for name, r in results.items():
        if r:
            lat_ms, fps = r
            print(f"  {name:<12} | GPU Lat. = {lat_ms}\\,ms  | GPU FPS = {fps}")
        else:
            print(f"  {name:<12} | NOT FOUND")
    print("=" * 60)
    print("\nCurrent values in paper (for reference):")
    print("  YOLOv8s  | 7.2 ms  | 139.3 FPS")
    print("  YOLOv11s | 34.3 ms | 29.1 FPS")
    print("  YOLOv12s | 28.2 ms | 35.5 FPS")
    print("  YOLOv26s | 24.2 ms | 41.3 FPS")


if __name__ == "__main__":
    main()
