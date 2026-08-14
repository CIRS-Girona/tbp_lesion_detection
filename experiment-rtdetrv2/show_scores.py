import json, os

DATA_ROOT = os.environ.get("RTDETRV2_OUTPUT", "/path/to/rtdetrv2_output")
RUNS = {
    "BASELINE @100": "rtdetrv2_r18vd_itobos_besthp_100e_1024",
    "SCALE    @100": "rtdetrv2_r18vd_itobos_scale_besthp_100e_1024",
}
for name, d in RUNS.items():
    p = os.path.join(DATA_ROOT, d, "log.txt")
    print("=" * 48)
    print(name)
    if not os.path.exists(p):
        print("  (no log yet)"); continue
    rows = [json.loads(l) for l in open(p) if l.strip()]
    if not rows:
        print("  (empty)"); continue
    print(f"  progress: epoch {rows[-1]['epoch']} / 99")
    print(f"  {'epoch':>6} {'mAP50':>8} {'mAP50-95':>9}")
    for r in rows[-6:]:
        v = r.get("test_coco_eval_bbox", [0, 0])
        print(f"  {r['epoch']:>6} {v[1]:>8.4f} {v[0]:>9.4f}")
    best = max(rows, key=lambda r: r.get("test_coco_eval_bbox", [0])[0])
    bv = best["test_coco_eval_bbox"]
    print(f"  >>> BEST so far: epoch {best['epoch']}  mAP50={bv[1]:.4f}  mAP50-95={bv[0]:.4f}")
print("=" * 48)
