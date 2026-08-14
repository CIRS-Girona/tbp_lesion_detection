#!/usr/bin/env python3
"""
D-FINE Ablation Training Wrapper
=================================
Avoids the YAML criterion format issue by patching criterion.weight_dict
AFTER D-FINE loads the model and criterion — not through YAML config.

Usage (mirrors D-FINE train.py arguments + loss overrides):
  python scripts/train_ablation_dfine.py \\
    -c results/configs/dfine_s_fullaug_sweep_100ep.yml \\
    --output-dir results/training/ablations/ablation_fgl0_50ep \\
    --seed 42 --use-amp \\
    --loss-fgl 0.0

This script:
  1. Loads D-FINE's config and solver exactly as train.py does
  2. Patches criterion.weight_dict for the specified keys
  3. Runs training normally
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DFINE_REPO = Path(os.environ.get("DFINE_REPO", "~/code/iToBoS/aritra/dfine_code/D-FINE")).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D-FINE ablation training with loss weight overrides")
    p.add_argument("-c", "--config",     type=Path, required=True)
    p.add_argument("--output-dir",       type=Path, default=None)
    p.add_argument("--seed",             type=int,  default=0)
    p.add_argument("--use-amp",          action="store_true")
    p.add_argument("--resume",           type=str,  default="")
    # Loss weight overrides — the only additions over train.py
    p.add_argument("--loss-fgl",  type=float, default=None, help="FGL loss weight (default 0.15)")
    p.add_argument("--loss-ddf",  type=float, default=None, help="DDF loss weight (default 1.5)")
    p.add_argument("--loss-vfl",  type=float, default=None, help="VFL loss weight (default 1.0)")
    p.add_argument("--loss-bbox", type=float, default=None, help="Bbox L1 loss weight (default 5.0)")
    p.add_argument("--loss-giou", type=float, default=None, help="GIoU loss weight (default 2.0)")
    return p.parse_args()


def patch_weight_dict(criterion, overrides: dict) -> None:
    """
    Patch criterion.weight_dict in-place after D-FINE instantiates it.

    D-FINE's DFINECriterion stores weight_dict as a plain Python dict attribute.
    We just update the values directly — no YAML involved.
    """
    if not hasattr(criterion, "weight_dict"):
        print(f"[WARN] criterion {type(criterion).__name__} has no weight_dict attribute")
        print(f"       Attributes: {[a for a in dir(criterion) if not a.startswith('_')]}")
        return

    original = dict(criterion.weight_dict)
    for key, new_val in overrides.items():
        if new_val is None:
            continue
        # Patch the main key
        if key in criterion.weight_dict:
            print(f"[ablation] {key}: {criterion.weight_dict[key]} → {new_val}")
            criterion.weight_dict[key] = new_val
        else:
            print(f"[ablation] {key}: (not found in weight_dict, adding) → {new_val}")
            criterion.weight_dict[key] = new_val

        # Also patch aux stage variants: loss_fgl_stg1, loss_fgl_stg2, etc.
        default_val = original.get(key, 1.0)
        ratio = new_val / default_val if default_val != 0 else 0.0
        for wk in list(criterion.weight_dict.keys()):
            if wk != key and (wk.startswith(key + "_") or
                               (wk.startswith(key.replace("loss_", "")) and "_stg" in wk)):
                old = criterion.weight_dict[wk]
                criterion.weight_dict[wk] = round(old * ratio, 6)
                print(f"[ablation]   {wk}: {old} → {criterion.weight_dict[wk]}  (proportional)")

    print(f"[ablation] Final weight_dict: {criterion.weight_dict}")


def main() -> None:
    args = parse_args()

    loss_overrides = {
        "loss_fgl":  args.loss_fgl,
        "loss_ddf":  args.loss_ddf,
        "loss_vfl":  args.loss_vfl,
        "loss_bbox": args.loss_bbox,
        "loss_giou": args.loss_giou,
    }
    has_overrides = any(v is not None for v in loss_overrides.values())

    sys.path.insert(0, str(DFINE_REPO))

    import torch

    try:
        from src.core import YAMLConfig  # type: ignore
    except ImportError as e:
        sys.exit(f"[ERROR] Cannot import D-FINE src. Check DFINE_REPO: {DFINE_REPO}\n{e}")

    if args.output_dir is not None:
        out_dir = args.output_dir.expanduser().resolve()
    else:
        out_dir = Path("./ablation_output").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ablation] output_dir: {out_dir}")

    # D-FINE caches output_dir during YAMLConfig.__init__, so we must write
    # a proper YAML file — patching yaml_cfg dict afterwards is too late.
    base_cfg_path = args.config.expanduser().resolve()
    try:
        import yaml as _yaml  # type: ignore
    except ImportError:
        sys.exit("[ERROR] PyYAML required: pip install pyyaml")

    with open(base_cfg_path, "r") as f:
        cfg_data = _yaml.safe_load(f)

    # Override keys we need before D-FINE reads them
    cfg_data["output_dir"]   = str(out_dir)
    cfg_data["use_amp"]      = args.use_amp
    cfg_data["seed"]         = args.seed
    cfg_data["clip_max_norm"] = 0.1   # gradient clipping: prevents NaN

    if args.resume:
        cfg_data["resume"] = args.resume

    # Write the patched config to the ablation output dir
    ablation_cfg_path = out_dir / "ablation_config.yml"
    with open(ablation_cfg_path, "w") as f:
        _yaml.safe_dump(cfg_data, f, sort_keys=False)
    print(f"[ablation] Config written to: {ablation_cfg_path}")

    resume_str = args.resume or ""
    cfg = YAMLConfig(str(ablation_cfg_path), resume=resume_str)

    # Sanity check output_dir
    actual_out = getattr(cfg, "output_dir", None) or cfg.yaml_cfg.get("output_dir", "?")
    print(f"[ablation] D-FINE output_dir confirmed: {actual_out}")
    if str(out_dir) not in str(actual_out):
        print(f"[WARN] output_dir mismatch! Expected: {out_dir}, Got: {actual_out}")
        print("[WARN] Checkpoints may go to the wrong folder.")

    try:
        from src.solver import DetSolver  # type: ignore
    except ImportError as e:
        sys.exit(f"[ERROR] Cannot import DetSolver: {e}")

    solver = DetSolver(cfg)

    # Patch weight_dict AFTER criterion is instantiated
    if has_overrides:
        print("\n[ablation] Patching criterion weight_dict...")

        criterion_obj = None
        for attr in ["criterion", "model"]:
            obj = getattr(solver, attr, None)
            if obj is not None and hasattr(obj, "weight_dict"):
                criterion_obj = obj
                break
            elif obj is not None and hasattr(obj, "criterion"):
                inner = getattr(obj, "criterion")
                if hasattr(inner, "weight_dict"):
                    criterion_obj = inner
                    break

        # Fallback: access via cfg
        if criterion_obj is None:
            try:
                criterion_obj = cfg.criterion
            except Exception:
                pass

        if criterion_obj is not None:
            patch_weight_dict(criterion_obj, loss_overrides)
            print("[ablation] weight_dict patch complete.\n")
        else:
            print("[WARN] Could not find criterion.weight_dict. Loss weights unchanged.")

    print("[ablation] Starting training...")
    solver.fit()
    print("[ablation] Training complete.")


if __name__ == "__main__":
    main()
