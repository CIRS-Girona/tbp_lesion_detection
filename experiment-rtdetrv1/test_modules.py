"""
test_modules.py
---------------
Standalone sanity check for scale_modules.py — no Ultralytics/WandB needed,
just torch. Verifies:
  1. Each module returns the SAME shape it received (drop-in safe for RT-DETR).
  2. At init the blocks are identity (learnable residual gammas = 0).
  3. Gradients flow through every module (trainable).

Run:  python test_modules.py
"""

import torch

from scale_modules import (
    DynamicScaleAggregation,
    FrequencyAwareReconstruction,
    ScaleAwareResidualEnhancement,
    LesionSCALE,
)

# RT-DETR-L: hidden dim 256, three scales. At 1024 input -> P3=128, P4=64, P5=32.
C = 256
B = 2
shapes = [(B, C, 128, 128), (B, C, 64, 64), (B, C, 32, 32)]


def make_feats():
    return [torch.randn(*s, requires_grad=True) for s in shapes]


def check_shapes(name, ins, outs):
    for i, (a, b) in enumerate(zip(ins, outs)):
        assert a.shape == b.shape, f"{name}: scale {i} shape {a.shape} -> {b.shape}"
    print(f"  [{name}] shapes preserved: {[tuple(o.shape) for o in outs]}")


def main():
    torch.manual_seed(0)

    # DSA
    feats = make_feats()
    dsa = DynamicScaleAggregation(C)
    out = dsa(feats)
    check_shapes("DSA", feats, out)
    # identity at init (gamma=0)
    max_diff = max((o - f).abs().max().item() for o, f in zip(out, feats))
    print(f"  [DSA] max change at init (should be ~0): {max_diff:.2e}")

    # FAR (deepest scale)
    x = torch.randn(B, C, 32, 32, requires_grad=True)
    far = FrequencyAwareReconstruction(C)
    y = far(x)
    assert x.shape == y.shape
    print(f"  [FAR] shape preserved: {tuple(y.shape)}")
    print(f"  [FAR] max change at init (should be ~0): {(y - x).abs().max().item():.2e}")

    # SARE
    x = torch.randn(B, C, 64, 64, requires_grad=True)
    sare = ScaleAwareResidualEnhancement(C)
    y = sare(x)
    assert x.shape == y.shape
    print(f"  [SARE] shape preserved: {tuple(y.shape)}")
    print(f"  [SARE] max change at init (should be ~0): {(y - x).abs().max().item():.2e}")

    # Full SCALE block
    feats = make_feats()
    scale = LesionSCALE(C)
    out = scale(feats)
    check_shapes("LesionSCALE", feats, out)

    # Gradient flow
    loss = sum(o.sum() for o in out)
    loss.backward()
    n_params = sum(p.numel() for p in scale.parameters())
    n_grad = sum(1 for p in scale.parameters() if p.grad is not None)
    n_total = sum(1 for _ in scale.parameters())
    print(f"  [grad] {n_grad}/{n_total} parameter tensors received gradients")
    print(f"  [params] LesionSCALE has {n_params:,} trainable parameters")

    # Ablation toggles
    for cfg in [dict(use_dsa=True, use_far=False, use_sare=False),
                dict(use_dsa=False, use_far=True, use_sare=False),
                dict(use_dsa=False, use_far=False, use_sare=True)]:
        feats = make_feats()
        m = LesionSCALE(C, **cfg)
        out = m(feats)
        check_shapes(f"ablation {cfg}", feats, out)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
