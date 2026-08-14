#!/usr/bin/env python
"""
hp_sweep_rtdetrv2.py
--------------------
Bayesian (Optuna TPE) hyper-parameter sweep for RT-DETRv2-S on iToBoS, built on
top of the official Baidu repo. Each trial fine-tunes from the COCO-pretrained
weights for a reduced epoch budget and is scored by validation mAP50-95.

Why Optuna: the official repo has no sweep harness. Optuna's TPE sampler IS
Bayesian optimization; with a shared sqlite study, multiple workers (one per GPU)
cooperate -> parallel + resumable.

Search space (RT-DETR loss gains are fixed, so we only tune optimization):
  lr               2e-5 .. 4e-4  (log)
  weight_decay     1e-5 .. 1e-3  (log)
  warmup_duration  250  .. 2000  (step 250)

USAGE (run FROM REPO ROOT rtdetrv2_pytorch/, venv active):
  pip install optuna
  # one worker per free GPU, all share sweep_out/study.db:
  CUDA_VISIBLE_DEVICES=1 python hp_sweep_rtdetrv2.py --n-trials 8 --epochs 24
  CUDA_VISIBLE_DEVICES=3 python hp_sweep_rtdetrv2.py --n-trials 8 --epochs 24
  # after they finish, print best + write a ready 72-epoch config:
  python hp_sweep_rtdetrv2.py --report

Outputs:
  sweep_out/study.db                              shared Optuna study
  sweep_out/trial_NNN/                            each trial's run (log.txt, *.pth)
  configs/rtdetrv2/rtdetrv2_r18vd_itobos_besthp.yml   (written by --report)

Author: Praveen Kumar Murali | June 2026
"""
import os
import sys
import json
import argparse
import subprocess

REPO = os.path.dirname(os.path.abspath(__file__))                 # repo root
BASE_CFG = os.path.join(REPO, "configs/rtdetrv2/rtdetrv2_r18vd_itobos.yml")
PRETRAINED_DEFAULT = "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
STUDY_DEFAULT = "rtdetrv2_s_hp"


def write_trial_cfg(cfg_path, out_dir, epochs, lr, wd, warmup):
    """Write a per-trial config that fully overrides the optimizer + warmup."""
    text = (
        f"__include__: ['{BASE_CFG}']\n"
        f"epoches: {epochs}\n"
        f"checkpoint_freq: 100000          # only last.pth + best.pth (save disk)\n"
        f"output_dir: {out_dir}\n"
        f"optimizer:\n"
        f"  type: AdamW\n"
        f"  params:\n"
        f"    - params: '^(?=.*(?:norm|bn)).*$'\n"
        f"      weight_decay: 0.0\n"
        f"  lr: {lr}\n"
        f"  betas: [0.9, 0.999]\n"
        f"  weight_decay: {wd}\n"
        f"lr_warmup_scheduler:\n"
        f"  type: LinearWarmup\n"
        f"  warmup_duration: {warmup}\n"
    )
    with open(cfg_path, "w") as f:
        f.write(text)


def best_map_from_log(log_txt, idx=0):
    """Best COCO stat across epochs. idx 0 = mAP50-95, idx 1 = mAP50."""
    if not os.path.exists(log_txt):
        return 0.0
    best = 0.0
    with open(log_txt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = d.get("test_coco_eval_bbox")
            if v:
                best = max(best, float(v[idx]))
    return best


def make_objective(args):
    def objective(trial):
        lr = trial.suggest_float("lr", 2e-5, 4e-4, log=True)
        wd = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        warmup = trial.suggest_int("warmup_duration", 250, 2000, step=250)

        out_dir = os.path.join(args.out_root, f"trial_{trial.number:03d}")
        os.makedirs(out_dir, exist_ok=True)
        cfg_path = os.path.join(out_dir, "config.yml")
        write_trial_cfg(cfg_path, out_dir, args.epochs, lr, wd, warmup)

        cmd = [sys.executable, "tools/train.py", "-c", cfg_path,
               "-t", args.pretrained, "--use-amp", "--seed", "0"]
        print(f"\n[trial {trial.number}] lr={lr:.2e} wd={wd:.2e} warmup={warmup} "
              f"epochs={args.epochs}\n  {' '.join(cmd)}", flush=True)

        run_log = os.path.join(out_dir, "train_stdout.log")
        with open(run_log, "w") as lf:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=REPO)

        score = best_map_from_log(os.path.join(out_dir, "log.txt"), idx=0)
        trial.set_user_attr("lr", lr)
        trial.set_user_attr("weight_decay", wd)
        trial.set_user_attr("warmup_duration", warmup)
        trial.set_user_attr("out_dir", out_dir)
        print(f"[trial {trial.number}] DONE  val mAP50-95 = {score:.4f}", flush=True)
        return score
    return objective


def report(args):
    import optuna
    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    done = [t for t in study.trials if t.value is not None]
    done.sort(key=lambda t: t.value, reverse=True)

    print(f"\n===== SWEEP REPORT ({len(done)} finished trials) =====")
    print(f"{'rank':>4} {'mAP50-95':>9} {'lr':>10} {'wd':>10} {'warmup':>7}")
    for i, t in enumerate(done):
        p = t.params
        print(f"{i:>4} {t.value:9.4f} {p['lr']:10.2e} {p['weight_decay']:10.2e} "
              f"{p['warmup_duration']:7d}")

    if not done:
        print("no finished trials yet."); return
    b = study.best_trial
    print(f"\nBEST: mAP50-95={b.value:.4f}  lr={b.params['lr']:.3e} "
          f"wd={b.params['weight_decay']:.3e} warmup={b.params['warmup_duration']}")

    final_cfg = os.path.join(REPO, "configs/rtdetrv2/rtdetrv2_r18vd_itobos_besthp.yml")
    out_dir = os.path.join(REPO, "output/rtdetrv2_r18vd_itobos_besthp_1024")
    write_trial_cfg(final_cfg, out_dir, 72,
                    b.params["lr"], b.params["weight_decay"], b.params["warmup_duration"])
    print(f"\nWrote 72-epoch config with best HPs -> {final_cfg}")
    print("Run the full training with:")
    print(f"  CUDA_VISIBLE_DEVICES=<gpu> python tools/train.py "
          f"-c configs/rtdetrv2/rtdetrv2_r18vd_itobos_besthp.yml "
          f"-t {args.pretrained} --use-amp --seed 0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=8, help="trials for THIS worker")
    ap.add_argument("--epochs", type=int, default=24, help="epochs per trial (fine-tune)")
    ap.add_argument("--out-root", default=os.path.join(REPO, "sweep_out"))
    ap.add_argument("--study-name", default=STUDY_DEFAULT)
    ap.add_argument("--storage", default=None, help="default sqlite in out-root")
    ap.add_argument("--pretrained", default=PRETRAINED_DEFAULT)
    ap.add_argument("--seed", type=int, default=None,
                    help="sampler seed; LEAVE UNSET for parallel workers so they diverge")
    ap.add_argument("--report", action="store_true", help="print best + write final cfg")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    if args.storage is None:
        args.storage = f"sqlite:///{os.path.join(args.out_root, 'study.db')}"

    import optuna
    if args.report:
        report(args); return

    # No fixed seed by default -> parallel workers explore DIFFERENT points.
    # n_startup_trials=5 so TPE (Bayesian) guidance kicks in after just 5 random
    # trials; multivariate models the lr/wd/warmup jointly.
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=5, multivariate=True, seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name, storage=args.storage,
        direction="maximize", load_if_exists=True, sampler=sampler,
    )
    study.optimize(make_objective(args), n_trials=args.n_trials)
    print("\nThis worker is done. Run `--report` once ALL workers finish.")


if __name__ == "__main__":
    main()
