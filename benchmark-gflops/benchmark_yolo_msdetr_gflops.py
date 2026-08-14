"""
benchmark_yolo_msdetr_gflops.py
-------------------------------
Computes GFLOPs (thop MACs) at 640x640 and 1024x1024 on the server.
Supports YOLOv8s, YOLOv11s, YOLOv12s, YOLOv26s, RT-DETR-L, and MS-DETR.

MS-DETR is run in an isolated subprocess to prevent import/package conflicts.

Usage on server:
  python benchmark_yolo_msdetr_gflops.py
"""

import os
import glob
import subprocess
import torch

try:
    from ultralytics import YOLO, RTDETR
except ImportError:
    print("Error: 'ultralytics' package is not installed in this environment.")
    exit(1)

try:
    import thop
except ImportError:
    print("Error: 'thop' package is not installed. Please run: pip install thop")
    exit(1)

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
    ]
    for d in search_dirs:
        if os.path.exists(os.path.join(d, "ultralytics/nn/tasks.py")):
            return d
    return None

def run_msdetr_gflops_subprocess(path):
    print("Measuring MS-DETR GFLOPs via isolated subprocess...")
    src_dir = find_msdetr_src()
    if not src_dir:
        print("  [ERROR] MS-DETR source directory (msdetr-src) not found on server.")
        return None

    helper_script = "run_msdetr_gflops_helper.py"
    cmd_code = f"""
import sys, os
from types import ModuleType

# Mock 'cpuinfo' module
cpuinfo_mock = ModuleType('cpuinfo')
cpuinfo_mock.get_cpu_info = lambda: {{'brand_raw': 'Intel Server CPU'}}
sys.modules['cpuinfo'] = cpuinfo_mock

sys.path.insert(0, '{src_dir}')
try:
    import torch
    import thop
    from ultralytics import YOLO
    model = YOLO('{path}')
    model.to('cpu')

    dummy_640 = torch.zeros(1, 3, 640, 640)
    macs_640, _ = thop.profile(model.model, inputs=(dummy_640,), verbose=False)
    gflops_640 = macs_640 / 1e9

    dummy_1024 = torch.zeros(1, 3, 1024, 1024)
    macs_1024, _ = thop.profile(model.model, inputs=(dummy_1024,), verbose=False)
    gflops_1024 = macs_1024 / 1e9

    print(f"RESULT|{{gflops_640:.2f}}|{{gflops_1024:.2f}}")
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
        stdout, stderr = process.communicate()

        if os.path.exists(helper_script):
            os.remove(helper_script)

        for line in stdout.splitlines():
            if line.startswith("RESULT|"):
                parts = line.split("|")
                g_640 = float(parts[1])
                g_1024 = float(parts[2])
                return {"640": g_640, "1024": g_1024}
            elif line.startswith("ERROR|"):
                print(f"  [ERROR in MS-DETR subprocess] {line}")
        if stderr:
            print(f"  [stderr] {stderr}")
        return None
    except Exception as e:
        print(f"  [ERROR] MS-DETR subprocess failed: {e}")
        if os.path.exists(helper_script):
            os.remove(helper_script)
        return None

def compute_gflops(name, path):
    if name == "MS-DETR":
        return run_msdetr_gflops_subprocess(path)

    print(f"Loading {name} for GFLOPs benchmark...")
    if not os.path.exists(path):
        print(f"  [SKIP] Weights not found: {path}")
        return None

    try:
        if name == "RT-DETR-L":
            model = RTDETR(path)
        else:
            model = YOLO(path)
        model.to('cpu')

        dummy_640 = torch.zeros(1, 3, 640, 640)
        macs_640, _ = thop.profile(model.model, inputs=(dummy_640,), verbose=False)
        g_640 = macs_640 / 1e9

        dummy_1024 = torch.zeros(1, 3, 1024, 1024)
        macs_1024, _ = thop.profile(model.model, inputs=(dummy_1024,), verbose=False)
        g_1024 = macs_1024 / 1e9

        print(f"  {name}: {g_640:.2f} GFLOPs @ 640px | {g_1024:.2f} GFLOPs @ 1024px")
        return {"640": g_640, "1024": g_1024}
    except Exception as e:
        print(f"  [ERROR] Failed GFLOPs for {name}: {e}")
        return None

def main():
    print("=" * 70)
    print("  GFLOPs Benchmark (YOLO, MS-DETR, RT-DETR-L)")
    print("  Device: CPU | Convention: thop MACs")
    print("=" * 70)

    results = {}
    for name, path in model_paths.items():
        res = compute_gflops(name, path)
        if res:
            results[name] = res

    print("\n" + "=" * 70)
    print("  SUMMARY TABLE (thop MACs GFLOPs)")
    print("=" * 70)
    print(f"  {'Model':<15} | {'GFLOPs @ 640px':<16} | {'GFLOPs @ 1024px':<16}")
    print("-" * 70)
    for name, res in results.items():
        print(f"  {name:<15} | {res['640']:>16.2f} | {res['1024']:>16.2f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
