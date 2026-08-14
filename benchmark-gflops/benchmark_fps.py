import os
"""
Inference Speed Benchmark — v11s, v12s, v26s
Measures: FPS, latency (ms/image), model parameters, model size
Run on the server with: python benchmark_fps.py
"""
import argparse, time, csv, glob
import torch
from ultralytics import YOLO

BASE_DIR = os.environ.get("ITOBOS_ROOT", "/path/to/itobos")

MODELS = {
    "yolo11s_full_aug": os.path.join(BASE_DIR, "yamin/experiment-v11/runs/best_model/yolo11s_best_full_100ep/weights/best.pt"),
    "yolo12s_full_aug": os.path.join(BASE_DIR, "yamin/experiment-v12/runs/best_model/yolo12s_best_full_100ep/weights/best.pt"),
    "yolo26s_full_aug": os.path.join(BASE_DIR, "yamin/runs/best_model/yolo26s_best_hparams_100ep_v1/weights/best.pt"),
}

TEST_IMAGES_DIR = os.environ.get("DATA_ROOT", "/path/to/itobos/dataset") + "/test/images"
IMG_SIZE    = 1024
WARMUP_RUNS = 20     # iterations to warm up GPU
BENCH_RUNS  = 100    # iterations to average over
CONF        = 0.20
DEVICE      = "1"
OUTPUT_CSV  = os.path.join(BASE_DIR, "yamin/benchmark_fps_results.csv")


def get_test_images(n=BENCH_RUNS + WARMUP_RUNS):
    imgs = sorted(glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg")) +
                  glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png")))
    if len(imgs) == 0:
        raise FileNotFoundError(f"No images found in {TEST_IMAGES_DIR}")
    # Cycle if fewer images than needed
    while len(imgs) < n:
        imgs = imgs + imgs
    return imgs[:n]


def count_params(model_path):
    """Return total trainable parameters (millions)."""
    m = YOLO(model_path)
    total = sum(p.numel() for p in m.model.parameters())
    return round(total / 1e6, 2)


def get_model_size_mb(model_path):
    return round(os.path.getsize(model_path) / (1024 * 1024), 1)


def benchmark_model(name, model_path, images):
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {name}")
    print(f"  Weights:      {model_path}")
    print(f"{'='*60}")

    if not os.path.exists(model_path):
        print(f"  [SKIP] Weights not found: {model_path}")
        return None

    model = YOLO(model_path)
    params_m = count_params(model_path)
    size_mb   = get_model_size_mb(model_path)

    warmup_imgs = images[:WARMUP_RUNS]
    bench_imgs  = images[WARMUP_RUNS:WARMUP_RUNS + BENCH_RUNS]

    print(f"  Parameters: {params_m}M | File size: {size_mb} MB")
    print(f"  Warming up ({WARMUP_RUNS} runs)...", end=" ", flush=True)

    for img in warmup_imgs:
        model.predict(img, imgsz=IMG_SIZE, conf=CONF, device=DEVICE,
                      verbose=False, save=False)
    print("done")

    if DEVICE != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"  Benchmarking ({BENCH_RUNS} runs)...", end=" ", flush=True)
    t_start = time.perf_counter()
    for img in bench_imgs:
        model.predict(img, imgsz=IMG_SIZE, conf=CONF, device=DEVICE,
                      verbose=False, save=False)

    if DEVICE != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()

    t_total = time.perf_counter() - t_start
    print("done")

    latency_ms = (t_total / BENCH_RUNS) * 1000
    fps        = 1000.0 / latency_ms

    print(f"  → Latency:  {latency_ms:.1f} ms/image")
    print(f"  → FPS:      {fps:.1f}")

    return {
        "model":        name,
        "weights":      model_path,
        "params_M":     params_m,
        "size_MB":      size_mb,
        "latency_ms":   round(latency_ms, 1),
        "fps":          round(fps, 1),
        "device":       "GPU A100" if DEVICE != "cpu" else "CPU",
        "img_size":     IMG_SIZE,
        "bench_runs":   BENCH_RUNS,
    }


def main():
    global DEVICE, BENCH_RUNS
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=DEVICE, help="cuda device (0,1,2) or cpu")
    parser.add_argument("--runs", type=int, default=BENCH_RUNS)
    args = parser.parse_args()

    DEVICE = args.device
    BENCH_RUNS = args.runs

    print(f"\niToBoS Inference Speed Benchmark")
    print(f"Device: {DEVICE} | Image size: {IMG_SIZE}x{IMG_SIZE} | Runs: {BENCH_RUNS}")

    images = get_test_images()
    print(f"Using {len(images)} test images from {TEST_IMAGES_DIR}")

    results = []
    for name, path in MODELS.items():
        r = benchmark_model(name, path, images)
        if r:
            results.append(r)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<25} {'Params':>8} {'Size':>8} {'Latency':>10} {'FPS':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    for r in results:
        print(f"  {r['model']:<25} {r['params_M']:>7}M {r['size_MB']:>7}MB "
              f"{r['latency_ms']:>9.1f}ms {r['fps']:>7.1f}")
    print(f"\n  Saved to: {OUTPUT_CSV}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
