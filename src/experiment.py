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

from . import train as train_mod
from . import validate as validate_mod
from .angles import angle_scale, energy_span
from .qaoa_ref import QAOA, P as DEPTH

REPO = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------------------
# The experiment. This dict is the knob — edit it, commit, re-run the notebook.
# --------------------------------------------------------------------------------------
CONFIG = {
    "name": "norm",         # names the run directory when not on Kaggle

    # training
    "train_hours": 6.0,     # wall-clock budget; the real control, not `iters`
    "iters": 12000,         # upper bound — the clock usually binds first
    "batch": 128,
    "steps": 8,             # rollout length
    "lr": 3e-4,
    "eval_every": 500,      # also the checkpoint interval
    "eval_restarts": 16,
    "seed": 0,

    # inference / validation
    "val_restarts": 256,
    "val_polish": 100,
    "val_top_m": 16,
}

# Applied on top of CONFIG by `--quick`: exercises every stage in a few minutes.
QUICK = {
    "name": "quick",
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
    """Where artefacts go. On Kaggle that must be /kaggle/working to be saved as output."""
    kaggle = Path("/kaggle/working")
    if kaggle.is_dir():
        return kaggle
    d = REPO / "runs" / cfg["name"]
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

    print("\n" + "-" * 78 + "\n[train]\n" + "-" * 78, flush=True)
    tr = train_mod.main([
        "--data-dir", str(args.data_dir), "--out-dir", str(out_dir),
        "--iters", str(cfg["iters"]), "--max-hours", str(cfg["train_hours"]),
        "--batch", str(cfg["batch"]), "--steps", str(cfg["steps"]), "--lr", str(cfg["lr"]),
        "--eval-every", str(cfg["eval_every"]), "--eval-restarts", str(cfg["eval_restarts"]),
        "--seed", str(cfg["seed"]), "--device", args.device,
    ])
    ckpt = Path(tr["best_ckpt"])
    if not ckpt.exists():
        raise RuntimeError("no checkpoint written — training died before the first eval")

    submission = out_dir / "submission_train.csv"
    print("\n" + "-" * 78 + "\n[validate]\n" + "-" * 78, flush=True)
    va = validate_mod.main([
        "--ckpt", str(ckpt), "--data-dir", str(args.data_dir),
        "--h", str(args.data_dir / "h_train.npy"),
        "--restarts", str(cfg["val_restarts"]), "--polish", str(cfg["val_polish"]),
        "--top-m", str(cfg["val_top_m"]), "--out", str(submission), "--device", args.device,
    ])

    print("\n" + "-" * 78 + "\n[summary]\n" + "-" * 78, flush=True)
    sm = summarise(submission, args.data_dir, args.device)

    summary = {"commit": git_sha(), "config": cfg, "scale": scale_info,
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
