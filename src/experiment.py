"""The experiment entry point: one run, start to finish, defined entirely in the repo.

The Kaggle notebook is a stub — it clones this repo and executes

    python -m src

and nothing else. Everything that decides *what a run does* lives here: the configuration
below, the stages in `main()`, and the reporting. Changing an experiment therefore means
editing this file (or the modules it calls) and pushing; the notebook never changes.

Stages
    preflight  device, data, and the measured angle scale — cheap checks worth failing on
    train      src.train, under a wall-clock budget (checkpoints survive an early stop)
    validate   src.validate, the full best-of-K + polish inference stack
    summary    re-score the written submission and compare against the reference numbers

Run it locally exactly the same way:
    python -m src                       # the configured run
    python -m src --quick               # ~5 min smoke test of the whole path
    python -m src --set train_hours=1 --set val_restarts=64
"""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from . import basins as basins_mod
from . import proposer as proposer_mod
from . import train as train_mod
from . import turbo as turbo_mod
from . import validate as validate_mod
from .angles import angle_scale, energy_span, to_angles
from .predict import write_submission
from .qaoa_ref import QAOA, P as DEPTH

REPO = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------------------
# The experiment. This dict is the knob — edit it, commit, re-run the notebook.
# --------------------------------------------------------------------------------------
CONFIG = {
    "name": "proposer",     # names the run directory when not on Kaggle
    "mode": "turbo",        # "turbo" = trust-region BO (no training stage at all);
                            # "proposer" = learned restarts; "rollout" = learned optimiser
    "seed": 0,

    # ---- mode="proposer" -------------------------------------------------------------
    # stage 1: label which random starts flow to the optimum (the expensive stage)
    "basin_hours": 3.0,     # wall-clock budget; partial output is kept and usable
    "basin_instances": 4000,
    "basin_starts": 256,    # random starts run per instance
    "basin_steps": 150,     # Adam steps per start
    "basin_rel_tol": 0.02,  # 'good' = final P within this of the instance's best
    "basin_init": "tqa",    # "uniform" = the old 10-D box; "tqa"/"ramp" = smooth schedules
    # stage 2: train h -> K starts against those labels
    "prop_iters": 6000,
    "prop_k": 10,           # starts proposed per instance
    "prop_batch": 256,
    "prop_lr": 1e-3,
    "prop_coverage": 1.0,   # weight on the diversity half of the Chamfer loss
    "prop_eval_every": 1000,
    # stage 3: inference — propose K, Adam from each, select after polishing
    "infer_polish": 300,

    # ---- mode="turbo" ----------------------------------------------------------------
    # Pure inference-time search: nothing is trained, so a run is one pass over h.
    # This is currently configured as a DIAGNOSTIC run: it deliberately exceeds the
    # 10-minute inference limit to find where the method saturates. `turbo_evals` at 400
    # and `turbo_hours` at 0 is the submission-legal setting.
    "turbo_evals": 6000,    # circuit evaluations per instance -- the real budget knob
    "turbo_hours": 1.5,     # wall-clock cap on the search; 0 = unlimited
    "turbo_n_tr": 2,        # independent trust regions per instance
    "turbo_n_cand": 512,    # Thompson candidates drawn per region per step
    "turbo_batch": 32,      # picks per region per step. Each step costs one GP
                            # factorisation whatever this is, so raising it buys
                            # evaluations far more cheaply than more steps do.
    "turbo_mem": 256,       # observations the local GP keeps per region
    "turbo_gp_dtype": "float32",   # a T4 runs float64 at 1/32 rate and the GP dominates
    "turbo_polish": 200,    # final Adam steps on TuRBO's best point
    "turbo_baselines": 4,   # matched-budget control runs (tqa and uniform)

    # ---- mode="rollout" --------------------------------------------------------------
    "train_hours": 6.0,     # wall-clock budget; the real control, not `iters`
    "iters": 12000,         # upper bound — the clock usually binds first
    "batch": 128,
    "steps": 8,             # rollout length
    "lr": 3e-4,
    "eval_every": 500,      # also the checkpoint interval
    "eval_restarts": 16,
    "val_restarts": 256,
    "val_polish": 100,
    "val_top_m": 16,
}

# Applied on top of CONFIG by `--quick`: exercises every stage in a few minutes.
QUICK = {
    "name": "quick",
    "turbo_evals": 60,
    "turbo_hours": 0.0,
    "turbo_n_tr": 1,
    "turbo_n_cand": 32,
    "turbo_batch": 4,
    "turbo_mem": 24,
    "turbo_polish": 20,
    "turbo_baselines": 2,
    "basin_hours": 0.05,
    "basin_instances": 32,
    "basin_starts": 16,
    "basin_steps": 40,
    "prop_iters": 200,
    "prop_eval_every": 100,
    "infer_polish": 40,
    "train_hours": 0.08,
    "iters": 200,
    "eval_every": 100,
    "eval_restarts": 4,
    "val_restarts": 16,
    "val_polish": 20,
}

# Mean P(ground) on h_train, for context in the final summary.
REFERENCE = {
    "transformer, pre-normalisation": 0.28165,
    "CMA-ES (nb 04)": 0.27725,
    "massive multistart (nb 03)": 0.32396,
    "leaderboard #10": 0.34882,
    "leaderboard #1": 0.81468,
}


def git_sha():
    try:
        r = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--pretty=%h %s"],
                           capture_output=True, text=True, timeout=15)
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def output_dir(cfg):
    """Where artefacts go, per host.

    Kaggle: /kaggle/working is the only directory saved as notebook output.
    Colab:  /content is the runtime's working directory and what the file browser opens
            on. The clone lives in /tmp, which is invisible there and dies with the
            runtime, so writing results next to the code would silently lose them.
    """
    if Path("/kaggle/working").is_dir():
        return Path("/kaggle/working")
    base = Path("/content") if Path("/content").is_dir() else REPO / "runs"
    d = base / cfg["name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def coerce(old, s):
    if isinstance(old, bool):
        return s.lower() in ("1", "true", "yes", "on")
    return type(old)(s) if old is not None else s


def banner(cfg, out_dir, device):
    print("=" * 78)
    print(f"commit   : {git_sha()}")
    print(f"repo     : {REPO}")
    print(f"outputs  : {out_dir}")
    print(f"device   : {device}   torch {torch.__version__}   {platform.node()}")
    if device == "cuda":
        print(f"gpu      : {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB)")
    else:
        print("gpu      : NONE — turn on the accelerator; training will not finish on CPU")
    print("config   : " + ", ".join(f"{k}={v}" for k, v in cfg.items()))
    print("=" * 78, flush=True)


def preflight(data_dir, device):
    """Fail fast on the things that make a six-hour run worthless.

    The measured angle scale is the one number worth reading before the GPU hours start:
    it is derived from the actual spectrum, not hardcoded (see angles.py). Expect a span
    near 39, a gamma unit near 0.16 rad, a beta unit of pi/2, and a ratio near 10x.
    """
    for f in ("J.npy", "h_train.npy"):
        if not (data_dir / f).exists():
            raise FileNotFoundError(f"{data_dir / f} is missing — is the clone complete?")

    sim = QAOA(np.load(data_dir / "J.npy"), device=device)
    h = torch.tensor(np.load(data_dir / "h_train.npy"), dtype=torch.float32, device=device)
    span = energy_span(sim, h)
    s = angle_scale(sim, h, device=device)
    info = {"instances": int(h.shape[0]), "energy_span": span,
            "gamma_unit": s[0].item(), "beta_unit": s[-1].item(),
            "ratio": s[-1].item() / s[0].item()}
    print(f"\n[preflight] h_train: {info['instances']} instances")
    print(f"[preflight] energy span {span:.3f}  ->  gamma unit {info['gamma_unit']:.4f} rad, "
          f"beta unit {info['beta_unit']:.4f} rad ({info['ratio']:.1f}x apart)", flush=True)
    del sim, h, s
    if device == "cuda":
        torch.cuda.empty_cache()
    return info


def summarise(submission, data_dir, device):
    """Re-score the file that was actually written, independently of the validate run."""
    rows = np.loadtxt(submission, delimiter=",", skiprows=1, dtype=np.float32)
    A = rows[:, 1:]
    sim = QAOA(np.load(data_dir / "J.npy"), device=device)
    h = torch.tensor(np.load(data_dir / "h_train.npy"), dtype=torch.float32, device=device)
    with torch.no_grad():
        p = sim.p_ground(h, torch.tensor(A[:, :DEPTH], device=device),
                         torch.tensor(A[:, DEPTH:], device=device)).cpu().numpy()
    distinct = len(np.unique(np.round(A, 6), axis=0))

    print(f"\nre-scored {submission.name}: mean {p.mean():.5f}  median {np.median(p):.5f}  "
          f"min {p.min():.5f}  max {p.max():.5f}")
    print(f"distinct rows: {distinct}/{len(A)}  (constant submissions score 0 by the rules)")
    for tag, ref in REFERENCE.items():
        print(f"  vs {tag:<32} {ref:.5f} : {p.mean() / ref:5.3f}x")
    g = A[:, :DEPTH]
    print(f"gamma actually used: |g| mean {np.abs(g).mean():.4f}, max {np.abs(g).max():.4f} rad")

    del sim, h
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"mean_p": float(p.mean()), "median_p": float(np.median(p)),
            "min_p": float(p.min()), "max_p": float(p.max()),
            "distinct_rows": distinct, "n": len(A)}


def stage(name):
    print("\n" + "-" * 78 + f"\n[{name}]\n" + "-" * 78, flush=True)


def validate_proposer(ckpt, data_dir, out, cfg, device):
    """Propose K starts, run Adam from each, select after polishing.

    Also prices the identical Adam budget spent on *random* starts. That comparison is
    the whole experiment: the classical optimiser is doing the optimising either way, so
    the only thing the model can contribute is a better place to begin.
    """
    model, scale, _ = proposer_mod.load_model(ckpt, device)
    sim = QAOA(np.load(data_dir / "J.npy"), device=device)
    h = torch.tensor(np.load(data_dir / "h_train.npy"), dtype=torch.float32, device=device)
    n, k = h.shape[0], model.k

    t0 = time.time()
    with torch.no_grad():
        u0 = model(h).transpose(0, 1)
    u, p = proposer_mod.polish_from(sim, h, u0, scale, cfg["infer_polish"], 0.03, 4096)
    secs = time.time() - t0
    print(f"  learned best-of-{k}: mean P = {p.mean().item():.4f}   "
          f"median = {p.median().item():.4f}   min = {p.min().item():.4f}   ({secs:.0f}s)",
          flush=True)

    r0 = proposer_mod.random_starts(k, n, device, cfg["seed"])
    _, pr = proposer_mod.polish_from(sim, h, r0, scale, cfg["infer_polish"], 0.03, 4096)
    lift = p.mean().item() / max(pr.mean().item(), 1e-12)
    print(f"   random best-of-{k}: mean P = {pr.mean().item():.4f}   "
          f"-> learned starts are {lift:.3f}x", flush=True)

    budget = "within" if secs < 600 else "OVER"
    print(f"\ninference time: {secs:.0f}s — {budget} the 10-minute budget")
    write_submission(out, to_angles(u, scale))
    return {"submission": out, "mean_p": p.mean().item(), "median_p": p.median().item(),
            "min_p": p.min().item(), "random_baseline": pr.mean().item(), "lift": lift,
            "seconds": secs, "within_budget": secs < 600}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="apply the QUICK preset")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE", help="override one CONFIG key (repeatable)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: /kaggle/working on Kaggle, else runs/<name>")
    ap.add_argument("--data-dir", type=Path, default=REPO / "data" / "raw")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    cfg = dict(CONFIG)
    if args.quick:
        cfg.update(QUICK)
    for kv in args.overrides:
        k, _, v = kv.partition("=")
        if k not in cfg:
            raise SystemExit(f"unknown config key {k!r}; known: {', '.join(sorted(cfg))}")
        cfg[k] = coerce(cfg[k], v)

    out_dir = args.out_dir or output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    banner(cfg, out_dir, args.device)

    scale_info = preflight(args.data_dir, args.device)
    submission = out_dir / "submission_train.csv"
    extra = {}

    if cfg["mode"] == "turbo":
        stage("turbo")
        va = turbo_mod.main([
            "--data-dir", str(args.data_dir), "--out", str(submission),
            "--evals", str(cfg["turbo_evals"]), "--n-tr", str(cfg["turbo_n_tr"]),
            "--max-hours", str(cfg["turbo_hours"]),
            "--n-cand", str(cfg["turbo_n_cand"]), "--polish", str(cfg["turbo_polish"]),
            "--batch", str(cfg["turbo_batch"]), "--mem", str(cfg["turbo_mem"]),
            "--gp-dtype", str(cfg["turbo_gp_dtype"]),
            "--baseline-starts", str(cfg["turbo_baselines"]),
            "--seed", str(cfg["seed"]), "--device", args.device,
        ])
        tr = {"note": "turbo has no training stage"}
        extra["turbo_no_polish"] = va["mean_p_no_polish"]
    elif cfg["mode"] == "proposer":
        npz = out_dir / "basins.npz"
        if npz.exists():
            print(f"\n[basins] reusing {npz} — delete it to regenerate", flush=True)
        else:
            stage("basins")
            basins_mod.main([
                "--data-dir", str(args.data_dir), "--out", str(npz),
                "--instances", str(cfg["basin_instances"]),
                "--starts", str(cfg["basin_starts"]), "--steps", str(cfg["basin_steps"]),
                "--rel-tol", str(cfg["basin_rel_tol"]),
                "--init", str(cfg["basin_init"]),
                "--max-hours", str(cfg["basin_hours"]),
                "--seed", str(cfg["seed"]), "--device", args.device,
            ])

        stage("proposer")
        tr = proposer_mod.main([
            "--basins", str(npz), "--data-dir", str(args.data_dir),
            "--out-dir", str(out_dir), "--k", str(cfg["prop_k"]),
            "--iters", str(cfg["prop_iters"]), "--batch", str(cfg["prop_batch"]),
            "--lr", str(cfg["prop_lr"]), "--coverage", str(cfg["prop_coverage"]),
            "--eval-every", str(cfg["prop_eval_every"]),
            "--polish-steps", str(cfg["infer_polish"]),
            "--init", str(cfg["basin_init"]),
            "--seed", str(cfg["seed"]), "--device", args.device,
        ])
        ckpt = Path(tr["best_ckpt"])
        if not ckpt.exists():
            raise RuntimeError("no proposer checkpoint — training died before the first eval")

        stage("validate")
        va = validate_proposer(ckpt, args.data_dir, submission, cfg, args.device)
        extra["random_baseline"] = va["random_baseline"]
    else:
        stage("train")
        tr = train_mod.main([
            "--data-dir", str(args.data_dir), "--out-dir", str(out_dir),
            "--iters", str(cfg["iters"]), "--max-hours", str(cfg["train_hours"]),
            "--batch", str(cfg["batch"]), "--steps", str(cfg["steps"]), "--lr", str(cfg["lr"]),
            "--eval-every", str(cfg["eval_every"]),
            "--eval-restarts", str(cfg["eval_restarts"]),
            "--seed", str(cfg["seed"]), "--device", args.device,
        ])
        ckpt = Path(tr["best_ckpt"])
        if not ckpt.exists():
            raise RuntimeError("no checkpoint written — training died before the first eval")

        stage("validate")
        va = validate_mod.main([
            "--ckpt", str(ckpt), "--data-dir", str(args.data_dir),
            "--h", str(args.data_dir / "h_train.npy"),
            "--restarts", str(cfg["val_restarts"]), "--polish", str(cfg["val_polish"]),
            "--top-m", str(cfg["val_top_m"]), "--out", str(submission),
            "--device", args.device,
        ])

    stage("summary")
    sm = summarise(submission, args.data_dir, args.device)

    summary = {"commit": git_sha(), "config": cfg, "scale": scale_info, **extra,
               "train": {k: (str(v) if isinstance(v, Path) else v) for k, v in tr.items()},
               "validate": {k: (str(v) if isinstance(v, Path) else v) for k, v in va.items()},
               "submission": sm, "total_seconds": time.time() - t0}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\ntotal {(time.time() - t0) / 60:.1f} min. artefacts in {out_dir}:")
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name:<24} {f.stat().st_size / 1024:8.0f} KiB")
    return summary


if __name__ == "__main__":
    main()
