#!/usr/bin/env python3
"""
WandB Bayesian sweep agent for D-FINE-S on iToBoS.

Usage (after creating the sweep):
  wandb agent <ENTITY>/<PROJECT>/<SWEEP_ID>

Or run a single trial manually:
  python scripts/sweep_agent.py --trial-lr 5e-5 --trial-wd 1e-4

The sweep tunes: lr, backbone_lr_ratio, weight_decay
for 30 epochs (fast proxy for full training).
The metric optimised: val mAP50 from D-FINE's log.txt.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DFINE_REPO   = Path(os.environ.get("DFINE_REPO", "~/code/iToBoS/aritra/dfine_code/D-FINE")).expanduser()
SCRIPTS_DIR  = Path(__file__).resolve().parent
RESULTS_DIR  = SCRIPTS_DIR / "results"
SWEEP_EPOCHS = 30
MODEL_SIZE   = "s"
BATCH        = 8
IMG_SIZE     = 1024
DEVICE       = "0"


def parse_best_map50(log_path: Path) -> float:
    """Parse best val mAP50 from D-FINE log.txt (JSONL format)."""
    best = 0.0
    if not log_path.exists():
        return best
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        # D-FINE logs: {"test_coco_eval_bbox": [AP, AP50, AP75, ...]}
        if "test_coco_eval_bbox" in row:
            ap_list = row["test_coco_eval_bbox"]
            if isinstance(ap_list, (list, tuple)) and len(ap_list) >= 2:
                best = max(best, float(ap_list[1]))
        # Fallback: some versions log mAP directly
        for key in ("val_mAP50", "mAP50", "AP50"):
            if key in row:
                best = max(best, float(row[key]))
    return best


def run_trial(lr: float, backbone_lr: float, weight_decay: float,
              run_name: str, use_wandb_in_dfine: bool = False) -> float:
    """Generate config and run D-FINE training. Returns best val mAP50."""
    make_cfg = str(SCRIPTS_DIR / "make_dfine_config_yamin.py")
    train_py  = str(DFINE_REPO / "train.py")
    output_dir = RESULTS_DIR / "sweep" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = RESULTS_DIR / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    # Generate YAML config
    cfg_args = [
        sys.executable, make_cfg,
        "--aug-mode", "full",
        "--model-size", MODEL_SIZE,
        "--epochs", str(SWEEP_EPOCHS),
        "--batch", str(BATCH),
        "--img-size", str(IMG_SIZE),
        "--results-dir", str(RESULTS_DIR),
        "--run-name", run_name,
        "--lr", str(lr),
        "--backbone-lr", str(backbone_lr),
        "--weight-decay", str(weight_decay),
    ]
    if use_wandb_in_dfine:
        cfg_args.append("--use-wandb")

    subprocess.run(cfg_args, check=True)
    config_path = config_dir / f"{run_name}.yml"

    # Launch D-FINE training
    cmd = [
        sys.executable, train_py,
        "-c", str(config_path),
        "--seed", "42",
        "--use-amp",
        "--output-dir", str(output_dir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = DEVICE
    env["PYTHONUNBUFFERED"] = "1"
    env["WANDB_MODE"] = "disabled"   # Disable WandB inside D-FINE; sweep agent logs separately.
    env["WANDB_DISABLED"] = "true"

    log_path = output_dir / "train_stdout.log"
    print(f"\n[sweep] Launching trial {run_name}")
    print(f"[sweep] lr={lr:.2e}  backbone_lr={backbone_lr:.2e}  wd={weight_decay:.2e}")

    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd, cwd=str(DFINE_REPO), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        ret = proc.wait()

    if ret != 0:
        print(f"[sweep] WARNING: training exited with code {ret}")

    dfine_log = output_dir / "log.txt"
    best_map50 = parse_best_map50(dfine_log)
    print(f"[sweep] Best val mAP50 = {best_map50:.4f}")
    return best_map50


def sweep_main() -> None:
    """Entry point when called by wandb agent."""
    try:
        import wandb
    except ImportError:
        raise SystemExit("wandb not installed. Run: pip install wandb")

    with wandb.init() as run:
        cfg = wandb.config
        lr          = float(cfg.lr)
        bb_ratio    = float(cfg.get("backbone_lr_ratio", 0.5))
        backbone_lr = lr * bb_ratio
        weight_decay= float(cfg.weight_decay)
        run_name    = f"sweep_{run.id}"

        best_map50 = run_trial(lr, backbone_lr, weight_decay, run_name)
        wandb.log({"val_mAP50": best_map50, "lr": lr, "weight_decay": weight_decay})


def manual_trial() -> None:
    """Run a single trial without WandB (for debugging)."""
    p = argparse.ArgumentParser()
    p.add_argument("--trial-lr",  type=float, default=5e-5)
    p.add_argument("--trial-bb",  type=float, default=None, help="backbone lr (default = lr/2)")
    p.add_argument("--trial-wd",  type=float, default=1e-4)
    p.add_argument("--run-name",  type=str,   default="manual_trial")
    args = p.parse_args()
    backbone_lr = args.trial_bb if args.trial_bb is not None else args.trial_lr / 2.0
    run_trial(args.trial_lr, backbone_lr, args.trial_wd, args.run_name)


if __name__ == "__main__":
    # wandb agent calls this script with sweep params via the sweep context;
    # wandb.init() inside sweep_main() reads them automatically.
    sweep_main()
