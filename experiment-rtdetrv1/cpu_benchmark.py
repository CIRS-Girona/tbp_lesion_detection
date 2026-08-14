"""
cpu_benchmark.py
----------------
CPU inference speed + GFLOPs benchmark for RT-DETR weights.
RT-DETR loads with the RTDETR class (not YOLO).

Usage:
  python cpu_benchmark.py --weights runs/best_model/rtdetr-l_best_full_100ep/weights/best.pt \
                          --image <one test image, 1024x1024>

Outputs: params, GFLOPs, mean CPU latency (ms), and FPS.
Protocol: 20 warmup iterations + 100 timed iterations on CPU at imgsz=1024.
"""

import argparse
import time

from ultralytics import RTDETR


def parse_args():
    p = argparse.ArgumentParser(description="RT-DETR CPU speed + GFLOPs benchmark")
    p.add_argument("--weights", type=str, required=True, help="path to best.pt")
    p.add_argument("--image", type=str, required=True, help="one test image (1024x1024)")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--runs", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    model = RTDETR(args.weights)

    print("\n=== Model info (params + GFLOPs) ===")
    model.info(detailed=False)

    print(f"\n=== CPU inference ({args.warmup} warmup + {args.runs} timed) ===")
    for _ in range(args.warmup):
        model.predict(args.image, device="cpu", imgsz=args.imgsz, verbose=False)

    times = []
    for _ in range(args.runs):
        t = time.time()
        model.predict(args.image, device="cpu", imgsz=args.imgsz, verbose=False)
        times.append(time.time() - t)

    mean_s = sum(times) / len(times)
    print(f"\nWeights : {args.weights}")
    print(f"Latency : {mean_s * 1000:.1f} ms (mean over {args.runs} runs)")
    print(f"FPS     : {1.0 / mean_s:.2f}")
    print("\n(Run all models the same way, same machine, same session, for fair comparison.)")


if __name__ == "__main__":
    main()
