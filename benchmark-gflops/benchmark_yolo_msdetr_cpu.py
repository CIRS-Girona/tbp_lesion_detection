"""
benchmark_yolo_msdetr_cpu.py
----------------------------
Controlled CPU inference speed benchmark for YOLOv8s, YOLOv11s, YOLOv12s, YOLOv26s, MS-DETR, and RT-DETR-L.
Runs 10 warm-up + 50 timed inferences on CPU using a dummy image to measure latency and FPS.

MS-DETR is run in a separate subprocess with its own custom ultralytics repository to prevent import/package conflicts.
Missing imports like 'cpuinfo' are dynamically mocked in the subprocess to ensure successful execution.

Usage on server:
  python benchmark_yolo_msdetr_cpu.py
"""

import time
import os
import glob
import numpy as np
import torch
import subprocess
import sys

try:
    from ultralytics import YOLO, RTDETR
except ImportError:
    print("Error: 'ultralytics' package is not installed in this environment.")
    exit(1)

IMGSZ  = 1024
WARMUP = 10
RUNS   = 50
DEVICE = "cpu"

BASE_DIR = os.environ.get("ITOBOS_ROOT", "/path/to/itobos")

model_paths = {
    "YOLOv8s":  os.path.join(BASE_DIR, "yamin/experiment-solomon-server/runs/best_model/solomon_yolov8s_full_100ep/weights/best.pt"),
    "YOLOv11s": os.path.join(BASE_DIR, "yamin/experiment-v11/runs/best_model/yolo11s_best_full_100ep/weights/best.pt"),
    "YOLOv12s": os.path.join(BASE_DIR, "yamin/experiment-v12/runs/best_model/yolo12s_best_full_100ep/weights/best.pt"),
    "YOLOv26s": os.path.join(BASE_DIR, "yamin/runs/best_model/yolo26s_best_hparams_100ep_v1/weights/best.pt"),
    "MS-DETR":  os.path.join(BASE_DIR, "yamin/experiment-msdetr/runs/best_model/msdetr_best_full_100ep/weights/best.pt"),
}

def find_yolov8s_path():
    p = model_paths["YOLOv8s"]
    if os.path.exists(p):
        return p
    guesses = [
        os.path.join(BASE_DIR, "yamin/experiment-solomon-server/runs/best_model/solomon_yolov8s_full_100ep/weights/best.pt"),
        os.path.join(BASE_DIR, "yamin/experiment-solomon/runs/best_model/solomon_yolov8s_full_100ep/weights/best.pt"),
    ]
    for g in guesses:
        if os.path.exists(g):
            return g
    # recursive fallback
    matches = glob.glob(os.path.join(BASE_DIR, "**/solomon_yolov8s_full_100ep/**/best.pt"), recursive=True)
    if matches:
        return matches[0]
    return p

model_paths["YOLOv8s"] = find_yolov8s_path()

def find_rtdetr_l_path():
    search_patterns = [
        os.path.join(BASE_DIR, "**/rtdetr-l_best_full_100ep/weights/best.pt"),
        os.path.join(BASE_DIR, "**/*rtdetr*/runs/best_model/*/weights/best.pt"),
        os.path.join(BASE_DIR, "**/experiment-rtdetr/**/weights/best.pt"),
    ]
    for pattern in search_patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    return None

rtdetr_l_path = find_rtdetr_l_path()
if rtdetr_l_path:
    model_paths["RT-DETR-L"] = rtdetr_l_path
else:
    model_paths["RT-DETR-L"] = os.path.join(BASE_DIR, "aritra/rtdetr_code/experiment-rtdetr/runs/best_model/rtdetr-l_best_full_100ep/weights/best.pt")

def find_msdetr_src():
    search_dirs = [
        os.path.join(BASE_DIR, "yamin/msdetr-src"),
        os.path.join(BASE_DIR, "yamin/source-code-ms-detr"),
        os.path.join(BASE_DIR, "yamin/source-code-ms-detr/Source Code/Source Code/MSDETR"),
    ]
    for d in search_dirs:
        if os.path.exists(os.path.join(d, "ultralytics/nn/tasks.py")):
            return d
    # Recursive search
    yamin_root = os.path.join(BASE_DIR, "yamin")
    if os.path.exists(yamin_root):
        for root, dirs, files in os.walk(yamin_root):
            if "tasks.py" in files and root.endswith("ultralytics/nn"):
                return os.path.dirname(os.path.dirname(root))
    return None

def benchmark_msdetr_subprocess(path):
    print("Benchmarking MS-DETR on CPU via isolated subprocess...")
    src_dir = find_msdetr_src()
    if not src_dir:
        print("  [ERROR] MS-DETR source directory (msdetr-src) not found on server.")
        return None
    print(f"  Using MS-DETR codebase: {src_dir}")

    helper_script = "run_msdetr_cpu_helper.py"
    cmd_code = f"""
import sys, time, os
from types import ModuleType

os.environ["WANDB_DISABLED"] = "true"
os.environ["YOLO_VERBOSE"] = "False"

# Mock 'cpuinfo' module
cpuinfo_mock = ModuleType('cpuinfo')
cpuinfo_mock.get_cpu_info = lambda: {{'brand_raw': 'Intel Server CPU'}}
sys.modules['cpuinfo'] = cpuinfo_mock

sys.path.insert(0, '{src_dir}')
try:
    print("STATUS|Loading MS-DETR model...")
    import numpy as np
    import torch
    from ultralytics import YOLO
    model = YOLO('{path}')
    model.to('cpu')
    print("STATUS|Model loaded successfully.")

    img = np.zeros(({IMGSZ}, {IMGSZ}, 3), dtype=np.uint8)

    print("STATUS|Running warm-up...")
    for i in range({WARMUP}):
        _ = model(img, device='cpu', verbose=False, imgsz={IMGSZ})
    print("STATUS|Warm-up complete.")

    print("STATUS|Running timed benchmark runs...")
    latencies = []
    for i in range({RUNS}):
        t0 = time.perf_counter()
        _ = model(img, device='cpu', verbose=False, imgsz={IMGSZ})
        latencies.append((time.perf_counter() - t0) * 1000)

    lat_mean = sum(latencies) / len(latencies)
    fps = 1000.0 / lat_mean
    print(f"RESULT|{{lat_mean:.2f}}|{{fps:.2f}}")
except Exception as e:
    print(f"ERROR|{{e}}")
"""
    try:
        with open(helper_script, "w") as f:
            f.write(cmd_code)
    except Exception as e:
        print(f"  [ERROR] Failed to write helper script: {e}")
        return None

    cmd = ["python", helper_script]
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        results_found = None
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                if line.startswith("STATUS|"):
                    print(f"  [Progress] {line.replace('STATUS|', '')}")
                elif line.startswith("RESULT|"):
                    parts = line.split("|")
                    lat_mean = float(parts[1])
                    fps = float(parts[2])
                    print(f"  → Latency: {lat_mean:.1f} ms")
                    print(f"  → FPS:     {fps:.1f}")
                    results_found = {
                        "model": "MS-DETR",
                        "latency_ms": round(lat_mean, 1),
                        "fps": round(fps, 1)
                    }
                elif line.startswith("ERROR|"):
                    print(f"  [ERROR] {line.replace('ERROR|', '')}")
                else:
                    print(f"  [stdout] {line}")

        stderr_output = process.stderr.read().strip()
        if stderr_output:
            print(f"  [stderr] {stderr_output}")

        process.wait()

        if os.path.exists(helper_script):
            os.remove(helper_script)

        return results_found
    except Exception as e:
        print(f"  [ERROR] Subprocess failed to execute: {e}")
        if os.path.exists(helper_script):
            os.remove(helper_script)
        return None

def benchmark_model(name, path):
    if name == "MS-DETR":
        return benchmark_msdetr_subprocess(path)

    print(f"\nBenchmarking {name} on CPU...")
    if not os.path.exists(path):
        print(f"  [SKIP] Weights file not found: {path}")
        return None

    try:
        if name == "RT-DETR-L":
            model = RTDETR(path)
        else:
            model = YOLO(path)
        model.to(DEVICE)
    except Exception as e:
        print(f"  [ERROR] Failed to load {name}: {e}")
        return None

    img = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)

    for _ in range(WARMUP):
        _ = model(img, device=DEVICE, verbose=False, imgsz=IMGSZ)

    latencies = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        _ = model(img, device=DEVICE, verbose=False, imgsz=IMGSZ)
        latencies.append((time.perf_counter() - t0) * 1000)

    lat_mean = np.mean(latencies)
    lat_std = np.std(latencies)
    fps = 1000.0 / lat_mean

    print(f"  → Latency: {lat_mean:.1f} ± {lat_std:.1f} ms")
    print(f"  → FPS:     {fps:.1f}")

    return {
        "model": name,
        "latency_ms": round(lat_mean, 1),
        "fps": round(fps, 1)
    }

def main():
    print("=" * 60)
    print("  CPU Inference Benchmark — YOLO, MS-DETR, RT-DETR-L")
    print(f"  Device: {DEVICE} | Resolution: {IMGSZ}x{IMGSZ} | Runs: {RUNS}")
    print("=" * 60)

    results = {}
    for name, path in model_paths.items():
        res = benchmark_model(name, path)
        if res:
            results[name] = res

    summary_lines = []
    summary_lines.append("\n" + "=" * 60)
    summary_lines.append("  SUMMARY TABLE")
    summary_lines.append("=" * 60)
    summary_lines.append(f"  {'Model':<15} | {'CPU Latency (ms)':<18} | {'CPU FPS':<10}")
    summary_lines.append("-" * 60)
    for name, res in results.items():
        summary_lines.append(f"  {name:<15} | {res['latency_ms']:>18.1f} | {res['fps']:>10.1f}")
    summary_lines.append("=" * 60)

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    output_txt = "benchmark_yolo_msdetr_cpu_results.txt"
    try:
        with open(output_txt, "w") as f:
            f.write(summary_text + "\n")
        print(f"Results saved to: {os.path.abspath(output_txt)}")
    except Exception as e:
        print(f"Error saving to txt file: {e}")

if __name__ == "__main__":
    main()
