#!/usr/bin/env python3
"""
Extended D-FINE config generator for the experiment-dfine folder.

Extends make_dfine_config.py with:
  - Configurable default paths via environment variables
  - Loss weight overrides for ablation studies
  - Reads criterion section from base config and patches weight_dict
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML required. Run: pip install pyyaml") from exc

DFINE_REPO   = Path(os.environ.get("DFINE_REPO", "~/D-FINE")).expanduser()
DATASET_DIR  = Path(os.environ.get("DFINE_DATASET_DIR", "~/dfine_results/dataset_coco")).expanduser()
RESULTS_DIR  = Path(os.environ.get("DFINE_RESULTS_DIR", "results")).expanduser()

# Per-model defaults (batch-size 8 → scale = 8/32 = 0.25)
MODEL_DEFAULTS = {
    "n": {"base_lr": 2.0e-4, "backbone_lr": 1.0e-4, "weight_decay": 1.0e-4},
    "s": {"base_lr": 2.0e-4, "backbone_lr": 1.0e-4, "weight_decay": 1.0e-4},
    "m": {"base_lr": 2.0e-4, "backbone_lr": 2.0e-5, "weight_decay": 1.0e-4},
    "l": {"base_lr": 2.5e-4, "backbone_lr": 1.25e-5, "weight_decay": 1.25e-4},
    "x": {"base_lr": 2.5e-4, "backbone_lr": 2.5e-6,  "weight_decay": 1.25e-4},
}

# D-FINE-S default loss weights (from paper Table S3 and official configs).
# These are overridden per-stage: stage-1 uses vfl+bbox+giou,
# stage-2 adds fgl+ddf per decoder layer. We expose them as top-level
# criterion.weight_dict keys; D-FINE sums aux-stage contributions
# automatically via the aux_weight_dict multiplier in DFINECriterion.
DFINE_LOSS_DEFAULTS = {
    "loss_vfl":  1.0,
    "loss_bbox": 5.0,
    "loss_giou": 2.0,
    "loss_fgl":  0.15,
    "loss_ddf":  1.5,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate D-FINE iToBoS config YAML")
    p.add_argument("--dfine-repo",    type=Path, default=DFINE_REPO)
    p.add_argument("--dataset-dir",   type=Path, default=DATASET_DIR)
    p.add_argument("--results-dir",   type=Path, default=RESULTS_DIR)
    p.add_argument("--model-size",    choices=["n", "s", "m", "l", "x"], default="s")
    p.add_argument("--aug-mode",      choices=["full", "noaug"], required=True)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch",         type=int,   default=8)
    p.add_argument("--val-batch",     type=int,   default=8)
    p.add_argument("--img-size",      type=int,   default=1024)
    p.add_argument("--workers",       type=int,   default=4)
    p.add_argument("--run-name",      type=str,   default=None)
    p.add_argument("--lr",            type=float, default=None)
    p.add_argument("--backbone-lr",   type=float, default=None)
    p.add_argument("--weight-decay",  type=float, default=None)
    p.add_argument("--use-wandb",     action="store_true")
    p.add_argument("--wandb-project", type=str,   default="skin-lesion-dfine-sweep")
    # Loss weight overrides for ablation studies
    p.add_argument("--loss-vfl",  type=float, default=None, help="Override criterion weight for VFL loss (default 1.0)")
    p.add_argument("--loss-bbox", type=float, default=None, help="Override criterion weight for L1 bbox loss (default 5.0)")
    p.add_argument("--loss-giou", type=float, default=None, help="Override criterion weight for GIoU loss (default 2.0)")
    p.add_argument("--loss-fgl",  type=float, default=None, help="Override criterion weight for FGL loss (default 0.15)")
    p.add_argument("--loss-ddf",  type=float, default=None, help="Override criterion weight for DDF loss (default 1.5)")
    return p.parse_args()


def resolve_includes(cfg_path: Path) -> Dict[str, Any]:
    """Recursively load a D-FINE YAML config, resolving __include__ chains."""
    text = cfg_path.read_text(encoding="utf-8")
    data: Dict[str, Any] = yaml.safe_load(text) or {}
    includes = data.pop("__include__", [])
    if isinstance(includes, str):
        includes = [includes]
    merged: Dict[str, Any] = {}
    for inc in includes:
        inc_path = Path(inc)
        if not inc_path.is_absolute():
            inc_path = cfg_path.parent / inc_path
        parent = resolve_includes(inc_path)
        _deep_merge(merged, parent)
    _deep_merge(merged, data)
    return merged


def _deep_merge(base: Dict, override: Dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def train_ops(img_size: int, aug_mode: str, stop_epoch: int) -> Dict:
    if aug_mode == "full":
        return {
            "type": "Compose",
            "ops": [
                {"type": "RandomPhotometricDistort", "p": 0.5},
                {"type": "RandomZoomOut", "fill": 0},
                {"type": "RandomIoUCrop", "p": 0.8},
                {"type": "SanitizeBoundingBoxes", "min_size": 1},
                {"type": "RandomHorizontalFlip"},
                {"type": "Resize", "size": [img_size, img_size]},
                {"type": "SanitizeBoundingBoxes", "min_size": 1},
                {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
                {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
            ],
            "policy": {
                "name": "stop_epoch",
                "epoch": stop_epoch,
                "ops": ["RandomPhotometricDistort", "RandomZoomOut", "RandomIoUCrop"],
            },
        }
    return {
        "type": "Compose",
        "ops": [
            {"type": "Resize", "size": [img_size, img_size]},
            {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
            {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
        ],
    }


def val_ops(img_size: int) -> Dict:
    return {
        "type": "Compose",
        "ops": [
            {"type": "Resize", "size": [img_size, img_size]},
            {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
        ],
    }


def build_criterion_override(args: argparse.Namespace) -> Optional[Dict]:
    """
    Build a criterion weight_dict override.

    D-FINE's YAML criterion field is a class-reference string (e.g. "DFINECriterion"),
    NOT a plain dict, so we cannot safely read it via yaml.safe_load.
    Instead we write ONLY the weight_dict override; D-FINE's __include__ deep-merge
    will keep all other criterion fields (type, num_classes, losses, etc.) from the
    base config untouched.
    """
    loss_overrides = {
        "loss_vfl":  args.loss_vfl,
        "loss_bbox": args.loss_bbox,
        "loss_giou": args.loss_giou,
        "loss_fgl":  args.loss_fgl,
        "loss_ddf":  args.loss_ddf,
    }
    if not any(v is not None for v in loss_overrides.values()):
        return None  # No overrides needed — skip criterion block entirely

    # Start from our known defaults (avoids reading the unparseable base YAML).
    weight_dict: Dict[str, float] = dict(DFINE_LOSS_DEFAULTS)

    for loss_key, new_val in loss_overrides.items():
        if new_val is None:
            continue
        weight_dict[loss_key] = float(new_val)

        # D-FINE uses aux stage keys: loss_fgl_stg1, loss_fgl_stg2, etc.
        # Scale them proportionally to the default so stg1/stg2 ratios are preserved.
        default_val = DFINE_LOSS_DEFAULTS.get(loss_key, 1.0)
        ratio = float(new_val) / default_val if default_val != 0 else 0.0
        aux_suffixes = ["_stg1", "_stg2", "_stg3", "_stg4", "_stg5", "_stg6"]
        for sfx in aux_suffixes:
            aux_key = loss_key + sfx
            if aux_key in DFINE_LOSS_DEFAULTS:
                weight_dict[aux_key] = round(DFINE_LOSS_DEFAULTS[aux_key] * ratio, 6)

    print(f"[criterion override] weight_dict patch: {weight_dict}")
    return {"weight_dict": weight_dict}


def build_config(args: argparse.Namespace) -> Path:
    dfine_repo  = args.dfine_repo.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()

    base_cfg = dfine_repo / "configs" / "dfine" / "custom" / f"dfine_hgnetv2_{args.model_size}_custom.yml"
    if not base_cfg.exists():
        raise FileNotFoundError(f"D-FINE base config not found: {base_cfg}")

    run_name   = args.run_name or f"dfine_{args.model_size}_{args.aug_mode}_{args.epochs}ep"
    output_dir = results_dir / "training" / args.aug_mode / run_name
    config_dir = results_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    defaults    = MODEL_DEFAULTS[args.model_size]
    scale       = args.batch / 32.0
    lr          = args.lr           if args.lr           is not None else defaults["base_lr"]    * scale
    backbone_lr = args.backbone_lr  if args.backbone_lr  is not None else defaults["backbone_lr"] * scale
    weight_decay= args.weight_decay if args.weight_decay is not None else defaults["weight_decay"]

    stop_epoch  = max(args.epochs - 2, 1) if args.aug_mode == "full" else args.epochs

    cfg: Dict[str, Any] = {
        "__include__": [str(base_cfg)],
        "output_dir":  str(output_dir),
        "print_freq":  50,
        "checkpoint_freq": 5,
        "sync_bn":     False,
        "find_unused_parameters": False,
        "use_amp":     True,
        "use_wandb":   bool(args.use_wandb),
        "project_name": args.wandb_project,
        "exp_name":    run_name,
        "num_classes": 1,
        "remap_mscoco_category": False,
        "epochs":      int(args.epochs),
        "eval_spatial_size": [args.img_size, args.img_size],
        "optimizer": {
            "type": "AdamW",
            "params": [
                {"params": "^(?=.*backbone)(?!.*norm|bn).*$",              "lr": float(backbone_lr)},
                {"params": "^(?=.*backbone)(?=.*norm|bn).*$",              "lr": float(backbone_lr), "weight_decay": 0.0},
                {"params": "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$", "weight_decay": 0.0},
            ],
            "lr":           float(lr),
            "betas":        [0.9, 0.999],
            "weight_decay": float(weight_decay),
        },
        "train_dataloader": {
            "total_batch_size": int(args.batch),
            "num_workers":      int(args.workers),
            "dataset": {
                "img_folder": str(dataset_dir / "images" / "train"),
                "ann_file":   str(dataset_dir / "annotations" / "instances_train.json"),
                "return_masks": False,
                "transforms":   train_ops(args.img_size, args.aug_mode, stop_epoch),
            },
            "collate_fn": {
                "type":             "BatchImageCollateFunction",
                "base_size":        int(args.img_size),
                "base_size_repeat": 3 if args.aug_mode == "full" else 1,
                "stop_epoch":       int(stop_epoch),
                "ema_restart_decay": 0.9999,
            },
        },
        "val_dataloader": {
            "total_batch_size": int(args.val_batch),
            "num_workers":      int(args.workers),
            "dataset": {
                "img_folder": str(dataset_dir / "images" / "val"),
                "ann_file":   str(dataset_dir / "annotations" / "instances_val.json"),
                "return_masks": False,
                "transforms":   val_ops(args.img_size),
            },
        },
    }

    criterion_override = build_criterion_override(args)
    if criterion_override is not None:
        cfg["criterion"] = criterion_override

    config_path = config_dir / f"{run_name}.yml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    meta = {
        "run_name": run_name, "aug_mode": args.aug_mode, "model_size": args.model_size,
        "epochs": args.epochs, "batch": args.batch, "img_size": args.img_size,
        "lr": lr, "backbone_lr": backbone_lr, "weight_decay": weight_decay,
        "loss_overrides": {k: getattr(args, k.replace("-", "_")) for k in
                           ["loss_vfl", "loss_bbox", "loss_giou", "loss_fgl", "loss_ddf"]},
        "output_dir": str(output_dir), "config_path": str(config_path),
    }
    (output_dir / "run_config_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Config  : {config_path}")
    print(f"Out dir : {output_dir}")
    return config_path


def main() -> None:
    build_config(parse_args())


if __name__ == "__main__":
    main()
