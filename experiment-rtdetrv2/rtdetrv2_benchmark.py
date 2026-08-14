#!/usr/bin/env python
"""
rtdetrv2_benchmark.py
Params + GFLOPs + CPU latency/FPS for the OFFICIAL RT-DETRv2-S baseline AND SCALE.
Run FROM THE REPO ROOT. GFLOPs needs thop (or calflops/fvcore); params+FPS need only torch.
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from src.core import YAMLConfig


def build_model(config, ckpt_path=None, device="cpu"):
    """Build the official RT-DETRv2 model (baseline OR SCALE) from config + ckpt."""
    cfg = YAMLConfig(config)
    model = cfg.model
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and ckpt.get("ema") is not None:
            state = ckpt["ema"]["module"]; tag = "EMA"
        elif isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]; tag = "model"
        else:
            state = ckpt; tag = "raw"
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[ckpt] loaded {tag} weights from {ckpt_path}")
        if missing:
            print(f"[ckpt] (info) {len(missing)} missing keys (usually buffers) - OK")
        if unexpected:
            print(f"[ckpt] (info) {len(unexpected)} unexpected keys - OK")
    return model.eval().to(device)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_gflops(model, imgsz, device="cpu"):
    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
    results = {}
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        results["thop_MACs(G)"] = macs / 1e9
        results["thop_2xMACs(G)"] = macs * 2 / 1e9
    except Exception as e:
        results["thop_error"] = str(e)[:80]
    try:
        from calflops import calculate_flops
        flops, macs, _ = calculate_flops(
            model=model, input_shape=(1, 3, imgsz, imgsz),
            output_as_string=False, print_results=False, print_detailed=False)
        results["calflops_FLOPs(G)"] = flops / 1e9
        results["calflops_MACs(G)"] = macs / 1e9
    except Exception as e:
        results["calflops_error"] = str(e)[:80]
    try:
        from fvcore.nn import FlopCountAnalysis
        fca = FlopCountAnalysis(model, dummy)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        results["fvcore_MACs(G)"] = fca.total() / 1e9
    except Exception as e:
        results["fvcore_error"] = str(e)[:80]
    return results


@torch.no_grad()
def measure_cpu_fps(model, imgsz, runs=50, warmup=10, device="cpu", threads=None):
    if threads is not None:
        torch.set_num_threads(int(threads))
    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
    for _ in range(warmup):
        model(dummy)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(dummy)
        times.append(time.perf_counter() - t0)
    times.sort()
    mean = sum(times) / len(times)
    median = times[len(times) // 2]
    return {"mean_ms": mean * 1e3, "median_ms": median * 1e3,
            "fps_mean": 1.0 / mean, "fps_median": 1.0 / median}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-r", "--resume", default=None)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-fps", action="store_true")
    args = ap.parse_args()

    print("=" * 64)
    print(f"  torch {torch.__version__} | device={args.device} | "
          f"threads={torch.get_num_threads()} | imgsz={args.imgsz}")
    print("=" * 64)

    model = build_model(args.config, args.resume, device=args.device)

    total, trainable = count_params(model)
    print(f"\n[PARAMS]  total = {total:,}  ({total/1e6:.2f} M)   "
          f"trainable = {trainable/1e6:.2f} M")

    print("\n[GFLOPs]  (input 1x3x{0}x{0})".format(args.imgsz))
    g = count_gflops(model, args.imgsz, device=args.device)
    if not g or all("error" in k for k in g):
        print("  no FLOP counter installed -> pip install thop")
    for k, v in g.items():
        if "error" in k:
            print(f"  {k:22s}: ({v})")
        else:
            print(f"  {k:22s}: {v:8.2f} G")
    # Ultralytics/YOLO 'GFLOPs' == the MAC count (thop_MACs / fvcore_MACs).
    print("  NOTE: report the SAME convention as the other table models.")

    if not args.no_fps:
        print(f"\n[CPU FPS]  batch=1, {args.runs} runs, {args.warmup} warmup "
              f"(forward only; RT-DETR is NMS-free)")
        f = measure_cpu_fps(model, args.imgsz, args.runs, args.warmup,
                            device=args.device, threads=args.threads)
        print(f"  latency: mean {f['mean_ms']:.1f} ms | median {f['median_ms']:.1f} ms")
        print(f"  FPS    : mean {f['fps_mean']:.2f}  | median {f['fps_median']:.2f}")

    print("\n" + "=" * 64)
    print("  DONE. Put params (M), GFLOPs (matching convention), CPU FPS in the table.")
    print("=" * 64)


if __name__ == "__main__":
    main()
