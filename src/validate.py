"""Validate a trained checkpoint on h_train with the full inference stack:
best-of-K parallel rollouts + optional Adam polish.

Reports mean/median/quantile P(ground) after each stage and wall-clock time,
so the result is directly comparable to the leaderboard metric and the
10-minute inference budget.

Usage:
    python -m src.validate                     # newest runs/**/best.pt
    python -m src.validate --ckpt runs/longer/best.pt --restarts 256 --polish 100
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .angles import to_angles
from .model import AngleTransformer
from .qaoa_ref import QAOA, P as DEPTH
from .predict import load_scale, polish, polish_then_select, write_submission
from .rollout import restart_candidates


def find_ckpt(runs_dir=Path("runs")):
    cands = sorted(runs_dir.glob("**/best.pt"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"no best.pt found under {runs_dir}/")
    return cands[-1]


def report(tag, p, t):
    q = torch.quantile(p, torch.tensor([0.1, 0.5, 0.9], device=p.device))
    print(
        f"{tag:>28}: mean P = {p.mean().item():.4f}   "
        f"median = {q[1].item():.4f}   p10 = {q[0].item():.4f}   p90 = {q[2].item():.4f}   "
        f"min = {p.min().item():.4f}   ({t:.0f}s elapsed)",
        flush=True,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=None, help="default: newest runs/**/best.pt")
    ap.add_argument("--h", type=Path, default=Path("data/raw/h_train.npy"))
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--restarts", type=int, default=256)
    ap.add_argument("--polish", type=int, default=100)
    ap.add_argument("--top-m", type=int, default=16, help="candidates polished per instance")
    ap.add_argument(
        "--out", type=Path, default=Path("runs/submission_train.csv"),
        help="write the final angles as a submission csv (main-stage leaderboard "
        "is scored on h_train, so this file is directly submittable)",
    )
    ap.add_argument("--steps", type=int, default=None, help="default: value stored in ckpt")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    device = args.device
    ckpt_path = args.ckpt or find_ckpt()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    steps = args.steps or cfg["steps"]
    print(
        f"checkpoint: {ckpt_path} (iter {ckpt.get('iter', '?')}, "
        f"trained eval metric {ckpt.get('metric', float('nan')):.4f})\n"
        f"h: {args.h}   restarts: {args.restarts}   steps: {steps}   "
        f"polish: {args.polish}   device: {device}",
        flush=True,
    )

    sim = QAOA(np.load(args.data_dir / "J.npy"), device=device)
    h = torch.tensor(np.load(args.h), dtype=torch.float32, device=device)
    model = AngleTransformer(
        d=cfg["d_model"], n_heads=cfg["heads"], n_layers=cfg["layers"],
        max_len=cfg["steps"] + 1,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    scale = load_scale(ckpt, device)

    t0 = time.time()
    cand_u, cand_p = restart_candidates(
        model, sim, h, steps, args.restarts, scale, chunk=args.chunk
    )
    p0, best_idx = cand_p.max(dim=0)
    report(f"best-of-{args.restarts} rollouts", p0, time.time() - t0)

    if args.polish > 0:
        # reference: old flow — polish only the single pre-polish best candidate
        cols = torch.arange(h.shape[0], device=device)
        u1 = polish(sim, h, cand_u[best_idx, cols], scale, args.polish)
        a1 = to_angles(u1, scale)
        p1 = sim.p_ground(h, a1[:, :DEPTH], a1[:, DEPTH:])
        report(f"polish best-1 x {args.polish}", p1, time.time() - t0)

        # polish the top-M candidates per instance, select AFTER polishing
        u, p = polish_then_select(sim, h, cand_u, cand_p, scale, args.polish, args.top_m)
        report(f"polish top-{args.top_m}, select after", p, time.time() - t0)
    else:
        cols = torch.arange(h.shape[0], device=device)
        u, p = cand_u[best_idx, cols], p0

    total = time.time() - t0
    budget = "within" if total < 600 else "OVER"
    print(f"\ntotal inference time: {total:.0f}s — {budget} the 10-minute budget")
    write_submission(args.out, to_angles(u, scale))     # csv is always in radians
    return {"submission": args.out, "mean_p": p.mean().item(),
            "median_p": p.median().item(), "min_p": p.min().item(),
            "seconds": total, "within_budget": total < 600}


if __name__ == "__main__":
    main()
